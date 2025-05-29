// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"

using namespace std::chrono;

// Constants
constexpr uint32_t WARMUP_ITERATIONS = 5;
constexpr uint32_t BENCHMARK_ITERATIONS = 20;
constexpr uint32_t CACHE_LINE_SIZE = 64;  // Typical cache line size in bytes
constexpr uint32_t PAGE_SIZE = 4096;      // Typical page size in bytes
constexpr uint32_t MIN_ARRAY_SIZE = 1024; // 1K elements
constexpr uint32_t MAX_ARRAY_SIZE = 67108864; // 64M elements

// Addressing modes to test
enum class AddressingMode {
  SEQUENTIAL,           // Sequential access (a[0], a[1], a[2], ...)
  STRIDED,              // Strided access (a[0], a[STRIDE], a[2*STRIDE], ...)
  RANDOM,               // Random access
  CACHE_LINE_ALIGNED,   // Aligned to cache line boundaries
  CACHE_LINE_UNALIGNED, // Unaligned to cache line boundaries
  PAGE_ALIGNED,         // Aligned to page boundaries
  PAGE_BOUNDARY_CROSS   // Deliberately cross page boundaries
};

// Helper function to get time in nanoseconds for more precise measurements
uint64_t get_time_ns() {
  return duration_cast<nanoseconds>(
             high_resolution_clock::now().time_since_epoch())
      .count();
}

// Template function to create access patterns based on the addressing mode
template <typename T>
std::vector<size_t> create_access_pattern(size_t size, AddressingMode mode,
                                          size_t stride = 16) {
  std::vector<size_t> indices(size);

  switch (mode) {
  case AddressingMode::SEQUENTIAL:
    // Simple sequential access
    for (size_t i = 0; i < size; i++) {
      indices[i] = i;
    }
    break;

  case AddressingMode::STRIDED:
    // Strided access pattern
    for (size_t i = 0; i < size; i++) {
      indices[i] = (i * stride) % size;
    }
    break;

  case AddressingMode::RANDOM: {
    // Random access pattern
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, size - 1);

    for (size_t i = 0; i < size; i++) {
      indices[i] = dist(gen);
    }
    break;
  }

  case AddressingMode::CACHE_LINE_ALIGNED: {
    // Access aligned to cache line boundaries
    size_t elements_per_line = CACHE_LINE_SIZE / sizeof(T);
    for (size_t i = 0; i < size; i++) {
      // Round down to cache line boundary, then add offset within line
      indices[i] = (i / elements_per_line) * elements_per_line;
    }
    break;
  }

  case AddressingMode::CACHE_LINE_UNALIGNED: {
    // Access deliberately unaligned to cache line boundaries
    size_t elements_per_line = CACHE_LINE_SIZE / sizeof(T);
    for (size_t i = 0; i < size; i++) {
      // Round down to cache line boundary, then add offset that crosses the
      // boundary
      indices[i] =
          (i / elements_per_line) * elements_per_line + elements_per_line / 2;
      if (indices[i] >= size) {
        indices[i] = size - 1;
      }
    }
    break;
  }

  case AddressingMode::PAGE_ALIGNED: {
    // Access aligned to page boundaries
    size_t elements_per_page = PAGE_SIZE / sizeof(T);
    for (size_t i = 0; i < size; i++) {
      // Round down to page boundary
      indices[i] = (i / elements_per_page) * elements_per_page;
    }
    break;
  }

  case AddressingMode::PAGE_BOUNDARY_CROSS: {
    // Deliberately cross page boundaries
    size_t elements_per_page = PAGE_SIZE / sizeof(T);
    for (size_t i = 0; i < size; i++) {
      // Access elements right at the page boundary
      indices[i] = (i / (elements_per_page / 2)) * elements_per_page - 1;
      if (indices[i] >= size) {
        indices[i] = size - 1;
      }
    }
    break;
  }
  }

  return indices;
}

// Structure to hold benchmark results
struct BenchmarkResult {
  std::string mode_name;
  size_t array_size;
  double avg_latency_ns;
  double bandwidth_mbps;
  double stddev_ns;
};

// Function to measure load/store instruction latencies for different addressing
// modes
void benchmark_memory_access_patterns() {
  std::cout << "\n===== Benchmarking Memory Access Patterns =====" << std::endl;

  struct TestCase {
    std::string name;
    AddressingMode mode;
    size_t stride;
  };

  std::vector<TestCase> test_cases = {
      {"Sequential", AddressingMode::SEQUENTIAL, 1},
      {"Strided-2", AddressingMode::STRIDED, 2},
      {"Strided-4", AddressingMode::STRIDED, 4},
      {"Strided-8", AddressingMode::STRIDED, 8},
      {"Strided-16", AddressingMode::STRIDED, 16},
      {"Random", AddressingMode::RANDOM, 1},
      {"CacheLine-Aligned", AddressingMode::CACHE_LINE_ALIGNED, 1},
      {"CacheLine-Unaligned", AddressingMode::CACHE_LINE_UNALIGNED, 1},
      {"Page-Aligned", AddressingMode::PAGE_ALIGNED, 1},
      {"Page-Boundary-Cross", AddressingMode::PAGE_BOUNDARY_CROSS, 1}};

  std::vector<size_t> array_sizes = {
      MIN_ARRAY_SIZE,      MIN_ARRAY_SIZE * 4,   MIN_ARRAY_SIZE * 16,
      MIN_ARRAY_SIZE * 64, MIN_ARRAY_SIZE * 256, MIN_ARRAY_SIZE * 1024,
      MAX_ARRAY_SIZE};

  std::vector<BenchmarkResult> results;

  // Use float data type for testing
  using DataType = float;

  for (const auto &size : array_sizes) {
    std::cout << "Testing array size: " << size << " elements ("
              << (size * sizeof(DataType) / (1024.0 * 1024.0)) << " MB)"
              << std::endl;

    // Create the data array
    std::vector<DataType> data(size, 1.0f);

    for (const auto &test : test_cases) {
      std::cout << "  Addressing mode: " << test.name << std::endl;

      // Create access pattern
      std::vector<size_t> indices =
          create_access_pattern<DataType>(size, test.mode, test.stride);

      // Warmup
      volatile DataType sum = 0;
      for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
        for (size_t j = 0; j < size; j++) {
          sum += data[indices[j]];
        }
      }

      // Benchmark runs - Read latency
      std::vector<double> read_times;
      for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
        auto start_time = get_time_ns();

        volatile DataType sum_read = 0;
        for (size_t j = 0; j < size; j++) {
          sum_read += data[indices[j]];
        }

        auto end_time = get_time_ns();
        double elapsed_ns = static_cast<double>(end_time - start_time);
        double ns_per_element = elapsed_ns / size;

        read_times.push_back(ns_per_element);
      }

      // Benchmark runs - Write latency
      std::vector<double> write_times;
      for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
        auto start_time = get_time_ns();

        for (size_t j = 0; j < size; j++) {
          data[indices[j]] = static_cast<DataType>(j % 100);
        }

        auto end_time = get_time_ns();
        double elapsed_ns = static_cast<double>(end_time - start_time);
        double ns_per_element = elapsed_ns / size;

        write_times.push_back(ns_per_element);
      }

      // Calculate average read latency
      double avg_read_latency =
          std::accumulate(read_times.begin(), read_times.end(), 0.0) /
          read_times.size();

      // Calculate standard deviation for read latency
      double read_variance = 0.0;
      for (double t : read_times) {
        read_variance += (t - avg_read_latency) * (t - avg_read_latency);
      }
      double read_stddev = std::sqrt(read_variance / read_times.size());

      // Calculate average write latency
      double avg_write_latency =
          std::accumulate(write_times.begin(), write_times.end(), 0.0) /
          write_times.size();

      // Calculate bandwidth in MB/s
      double read_bandwidth =
          sizeof(DataType) / (avg_read_latency * 1e-9) / (1024.0 * 1024.0);
      double write_bandwidth =
          sizeof(DataType) / (avg_write_latency * 1e-9) / (1024.0 * 1024.0);

      // Store results
      results.push_back({"READ-" + test.name, size, avg_read_latency,
                         read_bandwidth, read_stddev});
      results.push_back({"WRITE-" + test.name, size, avg_write_latency,
                         write_bandwidth, 0.0});

      std::cout << "    Read latency: " << avg_read_latency << " ns/element (±"
                << read_stddev << " ns)" << std::endl;
      std::cout << "    Read bandwidth: " << read_bandwidth << " MB/s"
                << std::endl;
      std::cout << "    Write latency: " << avg_write_latency << " ns/element"
                << std::endl;
      std::cout << "    Write bandwidth: " << write_bandwidth << " MB/s"
                << std::endl;
    }
  }

  // Write results to CSV
  std::ofstream result_file("memory_access_results.csv");
  result_file << "Mode,ArraySize,AverageLatency_ns,Bandwidth_MBps,StdDev_ns\n";

  for (const auto &result : results) {
    result_file << result.mode_name << "," << result.array_size << ","
                << result.avg_latency_ns << "," << result.bandwidth_mbps << ","
                << result.stddev_ns << "\n";
  }
  result_file.close();

  std::cout << "\nResults written to memory_access_results.csv" << std::endl;
}

// Function to detect cache line alignment effects
void benchmark_cache_line_effects() {
  std::cout << "\n===== Benchmarking Cache Line Alignment Effects ====="
            << std::endl;

  // Test different offsets from cache line boundary
  using DataType = float;
  size_t elements_per_line = CACHE_LINE_SIZE / sizeof(DataType);
  size_t array_size = MIN_ARRAY_SIZE * 64; // Use a moderately sized array

  std::vector<size_t> offsets;
  // Generate offsets from 0 to one complete cache line
  for (size_t i = 0; i <= elements_per_line; i++) {
    offsets.push_back(i);
  }

  std::cout << "Testing " << offsets.size() << " different alignments with "
            << elements_per_line << " elements per cache line" << std::endl;

  std::vector<std::pair<size_t, double>> alignment_results;

  // Create the data array
  std::vector<DataType> data(array_size + elements_per_line, 1.0f);

  for (size_t offset : offsets) {
    std::cout << "  Testing offset: " << offset
              << " elements from cache line boundary" << std::endl;

    // Create access pattern with the specific offset
    std::vector<size_t> indices(array_size);
    for (size_t i = 0; i < array_size; i++) {
      // Access at the given offset from each cache line boundary
      indices[i] = (i / elements_per_line) * elements_per_line + offset;
    }

    // Warmup
    volatile DataType sum = 0;
    for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
      for (size_t j = 0; j < array_size; j++) {
        sum += data[indices[j]];
      }
    }

    // Benchmark runs
    std::vector<double> times;
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
      auto start_time = get_time_ns();

      volatile DataType sum_local = 0;
      for (size_t j = 0; j < array_size; j++) {
        sum_local += data[indices[j]];
      }

      auto end_time = get_time_ns();
      double elapsed_ns = static_cast<double>(end_time - start_time);
      double ns_per_element = elapsed_ns / array_size;

      times.push_back(ns_per_element);
    }

    // Calculate average latency
    double avg_latency =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    // Store result
    alignment_results.push_back({offset, avg_latency});

    std::cout << "    Average latency: " << avg_latency << " ns/element"
              << std::endl;
  }

  // Write results to CSV
  std::ofstream result_file("cache_alignment_results.csv");
  result_file << "Offset,AverageLatency_ns\n";

  for (const auto &[offset, latency] : alignment_results) {
    result_file << offset << "," << latency << "\n";
  }
  result_file.close();

  std::cout << "\nResults written to cache_alignment_results.csv" << std::endl;
}

// Function to measure TLB miss rates by crossing page boundaries
void benchmark_tlb_effects() {
  std::cout << "\n===== Benchmarking TLB Miss Rates =====" << std::endl;

  using DataType = float;
  size_t elements_per_page = PAGE_SIZE / sizeof(DataType);

  // Test different numbers of pages to access
  std::vector<size_t> num_pages_list = {1,  2,   4,   8,   16,   32,
                                        64, 128, 256, 512, 1024, 2048};

  std::vector<std::tuple<size_t, double, double>> tlb_results;

  for (size_t num_pages : num_pages_list) {
    std::cout << "  Testing access across " << num_pages << " pages"
              << std::endl;

    size_t array_size = num_pages * elements_per_page;
    std::vector<DataType> data(array_size, 1.0f);

    // Create two access patterns:
    // 1. Sequential access within each page before moving to the next
    std::vector<size_t> sequential_indices(array_size);
    // 2. Strided access that crosses page boundaries
    std::vector<size_t> strided_indices(array_size);

    for (size_t i = 0; i < array_size; i++) {
      // For sequential: ((i / elements_per_page) * elements_per_page) + (i %
      // elements_per_page) This simplifies to just i, but it's explicitly
      // written out for clarity
      sequential_indices[i] = i;

      // For strided: jump between pages
      size_t page = i % num_pages;
      size_t offset = i / num_pages;
      if (offset < elements_per_page) {
        strided_indices[i] = page * elements_per_page + offset;
      } else {
        strided_indices[i] = i; // Fallback if we run out of offsets
      }
    }

    // Benchmark both patterns
    std::vector<std::string> pattern_names = {"Sequential", "Page-Crossing"};
    std::vector<std::vector<size_t>> patterns = {sequential_indices,
                                                 strided_indices};

    for (size_t p = 0; p < patterns.size(); p++) {
      const auto &indices = patterns[p];
      const auto &pattern_name = pattern_names[p];

      // Warmup
      volatile DataType sum = 0;
      for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
        for (size_t j = 0; j < array_size; j++) {
          sum += data[indices[j]];
        }
      }

      // Benchmark runs
      std::vector<double> times;
      for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
        auto start_time = get_time_ns();

        volatile DataType sum_local = 0;
        for (size_t j = 0; j < array_size; j++) {
          sum_local += data[indices[j]];
        }

        auto end_time = get_time_ns();
        double elapsed_ns = static_cast<double>(end_time - start_time);
        double ns_per_element = elapsed_ns / array_size;

        times.push_back(ns_per_element);
      }

      // Calculate average latency
      double avg_latency =
          std::accumulate(times.begin(), times.end(), 0.0) / times.size();

      // Store result
      tlb_results.push_back(
          {num_pages, p == 0 ? avg_latency : 0.0, p == 1 ? avg_latency : 0.0});

      std::cout << "    " << pattern_name << " access latency: " << avg_latency
                << " ns/element" << std::endl;
    }
  }

  // Write results to CSV
  std::ofstream result_file("tlb_effects_results.csv");
  result_file << "NumPages,SequentialLatency_ns,PageCrossingLatency_ns,Ratio\n";

  for (const auto &[num_pages, seq_latency, cross_latency] : tlb_results) {
    double ratio = cross_latency / seq_latency;
    result_file << num_pages << "," << seq_latency << "," << cross_latency
                << "," << ratio << "\n";
  }
  result_file.close();

  std::cout << "\nResults written to tlb_effects_results.csv" << std::endl;
}

int main(int argc, char **argv) {
  try {
    // Parse command line arguments
    CLI::App app{"Memory Access Pattern Benchmark for Tensix Hardware"};
    std::string benchmark_type = "all";
    app.add_option("-t,--type", benchmark_type,
                   "Benchmark type: 'access', 'cache', 'tlb', or 'all'")
        ->default_val(benchmark_type);

    CLI11_PARSE(app, argc, argv);

    // Run the requested benchmark(s)
    if (benchmark_type == "access" || benchmark_type == "all") {
      benchmark_memory_access_patterns();
    }

    if (benchmark_type == "cache" || benchmark_type == "all") {
      benchmark_cache_line_effects();
    }

    if (benchmark_type == "tlb" || benchmark_type == "all") {
      benchmark_tlb_effects();
    }

    std::cout << "Benchmarks completed." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}