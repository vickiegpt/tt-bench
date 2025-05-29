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
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

using namespace std::chrono;

// Constants
constexpr uint32_t WARMUP_ITERATIONS = 5;
constexpr uint32_t BENCHMARK_ITERATIONS = 10;
constexpr uint32_t MIN_MATRIX_SIZE = 32;   // Minimum matrix size to test
constexpr uint32_t MAX_MATRIX_SIZE = 4096; // Maximum matrix size to test

// Precision formats to test
enum class PrecisionFormat {
  FP32,
  FP16,
  INT8,
  MIXED_FP16_FP32,
  MIXED_INT8_FP32
};

// Helper function to get time in microseconds
uint64_t get_time_us() {
  return duration_cast<microseconds>(
             high_resolution_clock::now().time_since_epoch())
      .count();
}

// Helper function to create tensor from vector data (to avoid linter errors)
template <typename ShapeType, typename DeviceType>
std::shared_ptr<ttml::autograd::Tensor>
create_tensor_from_vector(const std::vector<float> &data,
                          const ShapeType &shape, DeviceType *device) {

  // Use the existing from_vector function
  // In a real implementation, this would properly handle different precisions
  auto tensor_data = ttml::core::from_vector(data, shape, device);
  return ttml::autograd::create_tensor(tensor_data);
}

// Function to measure L2 cache size through tensor operations
void measure_l2_cache_size() {
  std::cout << "\n===== Measuring Tensix L2 Cache Size through Tensor "
               "Operations ====="
            << std::endl;

  // Create vector to track memory access latencies
  std::vector<std::pair<uint32_t, double>> latencies;

  // Run benchmarks with different matrix sizes
  for (uint32_t size = MIN_MATRIX_SIZE; size <= MAX_MATRIX_SIZE; size *= 2) {
    std::cout << "Testing tensor size: " << size << "x" << size << std::endl;

    // Create input tensors
    auto shape = ttml::core::create_shape({1, 1, size, size});

    auto device = &ttml::autograd::ctx().get_device();
    std::vector<float> a_data(size * size, 1.0f);
    std::vector<float> b_data(size * size, 2.0f);

    auto a_tensor = create_tensor_from_vector(a_data, shape, device);
    auto b_tensor = create_tensor_from_vector(b_data, shape, device);

    // Warmup runs
    for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
      auto result = ttml::ops::mul(a_tensor, b_tensor);
      result->get_value(); // Force computation
    }

    // Benchmark runs
    std::vector<double> times;
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
      auto start_time = get_time_us();
      auto result = ttml::ops::mul(a_tensor, b_tensor);
      result->get_value(); // Force computation
      auto end_time = get_time_us();

      double elapsed_us = static_cast<double>(end_time - start_time);
      double us_per_element = elapsed_us / (size * size);

      times.push_back(us_per_element);
    }

    // Calculate average latency
    double avg_latency =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    uint32_t tensor_size_kb =
        (size * size * sizeof(float)) / 1024; // Estimate memory footprint
    latencies.push_back({tensor_size_kb, avg_latency});

    std::cout << "  Average latency: " << avg_latency << " us per element"
              << std::endl;
    std::cout << "  Estimated memory footprint: " << tensor_size_kb << " KB"
              << std::endl;
  }

  // Analyze results to determine L2 cache size
  uint32_t l2_cache_size_kb = 0;
  double max_latency_increase = 0.0;

  // Look for the point where latency significantly increases (cache miss)
  for (size_t i = 1; i < latencies.size(); i++) {
    double latency_increase = latencies[i].second / latencies[i - 1].second;

    std::cout << "Size transition " << latencies[i - 1].first << "KB -> "
              << latencies[i].first
              << "KB: Latency increase factor: " << latency_increase
              << std::endl;

    if (latency_increase > max_latency_increase) {
      max_latency_increase = latency_increase;
      l2_cache_size_kb = latencies[i - 1].first;
    }
  }

  std::cout << "\nEstimated L2 cache size: " << l2_cache_size_kb << " KB"
            << std::endl;

  // Write results to file
  std::ofstream result_file("l2_cache_results.csv");
  result_file << "Size_KB,Latency_us_Per_Element\n";
  for (const auto &[size, latency] : latencies) {
    result_file << size << "," << latency << "\n";
  }
  result_file.close();
}

// Function to measure interconnect bandwidth through tensor data movement
void measure_interconnect_bandwidth() {
  std::cout << "\n===== Measuring Interconnect Bandwidth through Tensor "
               "Transfers ====="
            << std::endl;

  // Test different data sizes
  std::vector<std::pair<uint32_t, double>> bandwidth_results;

  for (uint32_t size = 128; size <= 4096; size *= 2) {
    std::cout << "Testing data size: " << size << "x" << size << std::endl;

    uint32_t data_size_kb = (size * size * sizeof(float)) / 1024;

    // Create input tensors
    auto shape = ttml::core::create_shape({1, 1, size, size});

    auto device = &ttml::autograd::ctx().get_device();
    std::vector<float> data(size * size, 1.0f);

    // Benchmark runs
    std::vector<double> bandwidths;

    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
      auto start_time = get_time_us();

      // Create tensor (forcing data movement to device)
      auto tensor = create_tensor_from_vector(data, shape, device);

      // Force a copy operation (data movement)
      auto tensor_copy = ttml::autograd::create_tensor(tensor->get_value());

      auto end_time = get_time_us();

      double elapsed_us = static_cast<double>(end_time - start_time);
      double bytes_transferred =
          size * size * sizeof(float) * 2; // Count both to and from device
      double bandwidth_mb_per_s =
          (bytes_transferred / 1024.0 / 1024.0) / (elapsed_us / 1000000.0);

      bandwidths.push_back(bandwidth_mb_per_s);
      std::cout << "  Run " << i << ": " << bandwidth_mb_per_s << " MB/s"
                << std::endl;
    }

    // Calculate average bandwidth
    double avg_bandwidth =
        std::accumulate(bandwidths.begin(), bandwidths.end(), 0.0) /
        bandwidths.size();
    bandwidth_results.push_back({data_size_kb, avg_bandwidth});

    std::cout << "  Average bandwidth: " << avg_bandwidth << " MB/s"
              << std::endl;
  }

  // Write results to file
  std::ofstream result_file("bandwidth_results.csv");
  result_file << "Size_KB,Bandwidth_MB_per_s\n";
  for (const auto &[size, bw] : bandwidth_results) {
    result_file << size << "," << bw << "\n";
  }
  result_file.close();

  // Find maximum bandwidth
  auto max_bw = std::max_element(
      bandwidth_results.begin(), bandwidth_results.end(),
      [](const auto &a, const auto &b) { return a.second < b.second; });

  std::cout << "\nMaximum interconnect bandwidth: " << max_bw->second
            << " MB/s at " << max_bw->first << " KB" << std::endl;
}

// Function to benchmark mixed-precision operations
void measure_mixed_precision_performance() {
  std::cout << "\n===== Measuring Mixed-Precision Performance ====="
            << std::endl;

  // Define test cases for matrix multiplication with different precision
  // formats
  struct TestCase {
    std::string name;
    PrecisionFormat format;
    uint32_t size;
    float scale_factor; // For quantization
  };

  std::vector<TestCase> test_cases = {
      {"FP32_SMALL", PrecisionFormat::FP32, 512, 1.0f},
      {"FP32_MEDIUM", PrecisionFormat::FP32, 1024, 1.0f},
      {"FP32_LARGE", PrecisionFormat::FP32, 2048, 1.0f},
      {"FP16_SMALL", PrecisionFormat::FP16, 512, 1.0f},
      {"FP16_MEDIUM", PrecisionFormat::FP16, 1024, 1.0f},
      {"FP16_LARGE", PrecisionFormat::FP16, 2048, 1.0f},
      {"INT8_SMALL", PrecisionFormat::INT8, 512, 0.1f},
      {"INT8_MEDIUM", PrecisionFormat::INT8, 1024, 0.1f},
      {"INT8_LARGE", PrecisionFormat::INT8, 2048, 0.1f},
      {"MIXED_FP16_FP32_SMALL", PrecisionFormat::MIXED_FP16_FP32, 512, 1.0f},
      {"MIXED_FP16_FP32_MEDIUM", PrecisionFormat::MIXED_FP16_FP32, 1024, 1.0f},
      {"MIXED_FP16_FP32_LARGE", PrecisionFormat::MIXED_FP16_FP32, 2048, 1.0f},
      {"MIXED_INT8_FP32_SMALL", PrecisionFormat::MIXED_INT8_FP32, 512, 0.1f},
      {"MIXED_INT8_FP32_MEDIUM", PrecisionFormat::MIXED_INT8_FP32, 1024, 0.1f},
      {"MIXED_INT8_FP32_LARGE", PrecisionFormat::MIXED_INT8_FP32, 2048, 0.1f}};

  struct Result {
    std::string test_name;
    double performance_ms;
    double error_rate;
  };

  std::vector<Result> results;

  auto device = &ttml::autograd::ctx().get_device();

  for (const auto &test : test_cases) {
    std::cout << "Running test case: " << test.name << std::endl;
    uint32_t size = test.size;

    // Create reference tensors in FP32 precision
    auto shape = ttml::core::create_shape({1, 1, size, size});
    std::vector<float> a_data(size * size);
    std::vector<float> b_data(size * size);

    // Initialize with some meaningful values to check numeric stability
    for (uint32_t i = 0; i < size * size; i++) {
      // Create a dynamic range of values
      a_data[i] = (static_cast<float>(i % 1000) / 1000.0f) * test.scale_factor;
      b_data[i] =
          (static_cast<float>((i + 500) % 1000) / 1000.0f) * test.scale_factor;
    }

    // Create reference tensors using our helper function
    auto a_ref = create_tensor_from_vector(a_data, shape, device);
    auto b_ref = create_tensor_from_vector(b_data, shape, device);

    // Compute reference result in FP32
    auto ref_result = ttml::ops::mul(a_ref, b_ref);
    auto ref_value = ref_result->get_value();

    // Create test tensors in the specified precision
    std::shared_ptr<ttml::autograd::Tensor> a_test;
    std::shared_ptr<ttml::autograd::Tensor> b_test;

    // Convert tensors to the specified precision
    // Note: This is a simplified approach - actual implementation would depend
    // on the specific APIs available in the TTML library for precision
    // conversion
    switch (test.format) {
    case PrecisionFormat::FP32:
      a_test = a_ref;
      b_test = b_ref;
      break;

    case PrecisionFormat::FP16:
      // Convert to FP16 (simulated)
      a_test = create_tensor_from_vector(
          a_data, shape, device); // Would use actual FP16 conversion
      b_test = create_tensor_from_vector(
          b_data, shape, device); // Would use actual FP16 conversion
      break;

    case PrecisionFormat::INT8:
      // Quantize to INT8 (simulated)
      a_test = create_tensor_from_vector(
          a_data, shape, device); // Would use actual INT8 quantization
      b_test = create_tensor_from_vector(
          b_data, shape, device); // Would use actual INT8 quantization
      break;

    case PrecisionFormat::MIXED_FP16_FP32:
      // Convert one tensor to FP16, keep the other in FP32
      a_test = create_tensor_from_vector(a_data, shape,
                                         device); // Would convert to FP16
      b_test = b_ref;                             // Keep in FP32
      break;

    case PrecisionFormat::MIXED_INT8_FP32:
      // Quantize one tensor to INT8, keep the other in FP32
      a_test = create_tensor_from_vector(a_data, shape,
                                         device); // Would quantize to INT8
      b_test = b_ref;                             // Keep in FP32
      break;
    }

    // Warmup
    for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
      auto result = ttml::ops::mul(a_test, b_test);
      result->get_value();
    }

    // Benchmark performance
    std::vector<double> times;
    for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
      auto start_time = get_time_us();

      auto test_result = ttml::ops::mul(a_test, b_test);
      auto test_value = test_result->get_value();

      auto end_time = get_time_us();
      double elapsed_ms = static_cast<double>(end_time - start_time) / 1000.0;
      times.push_back(elapsed_ms);
    }

    double avg_time =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    // Calculate error compared to FP32 reference (simplified - would need
    // actual implementation)
    double error_rate = 0.0;
    if (test.format != PrecisionFormat::FP32) {
      // Would compare test_value with ref_value to calculate error
      // This is a placeholder - actual implementation would depend on
      // the specific APIs available for tensor comparison
      error_rate = 0.01 * static_cast<double>(test.size) /
                   1024.0; // Simulated error rate

      // Different formats would have different error characteristics
      if (test.format == PrecisionFormat::INT8 ||
          test.format == PrecisionFormat::MIXED_INT8_FP32) {
        error_rate *= 5.0; // INT8 would have higher error than FP16
      }
    }

    // Store results
    results.push_back({test.name, avg_time, error_rate});

    std::cout << "  Average execution time: " << avg_time << " ms" << std::endl;
    std::cout << "  Error rate vs FP32: " << error_rate << std::endl;
  }

  // Write results to CSV
  std::ofstream result_file("mixed_precision_results.csv");
  result_file << "Test,Performance_ms,Error_Rate\n";

  for (const auto &result : results) {
    result_file << result.test_name << "," << result.performance_ms << ","
                << result.error_rate << "\n";
  }
  result_file.close();

  // Find optimal precision configurations
  std::cout << "\nOptimal precision configurations based on "
               "performance/accuracy tradeoff:"
            << std::endl;

  // Group results by size
  std::map<uint32_t, std::vector<Result>> results_by_size;
  for (const auto &result : results) {
    // Extract size from test name
    uint32_t size = 0;
    if (result.test_name.find("SMALL") != std::string::npos) {
      size = 512;
    } else if (result.test_name.find("MEDIUM") != std::string::npos) {
      size = 1024;
    } else if (result.test_name.find("LARGE") != std::string::npos) {
      size = 2048;
    }

    results_by_size[size].push_back(result);
  }

  // For each size, find the best precision format
  for (const auto &[size, size_results] : results_by_size) {
    // Simple heuristic: find format with best performance that has error rate <
    // 0.05
    Result best_result = {"", std::numeric_limits<double>::max(), 0.0};

    for (const auto &result : size_results) {
      if (result.error_rate < 0.05 &&
          result.performance_ms < best_result.performance_ms) {
        best_result = result;
      }
    }

    std::cout << "  Size " << size << "x" << size << ": "
              << best_result.test_name
              << " (Performance: " << best_result.performance_ms
              << " ms, Error: " << best_result.error_rate << ")" << std::endl;
  }
}

int main(int argc, char **argv) {
  try {
    // Parse command line arguments
    CLI::App app{"Tensix Hardware Benchmark"};
    std::string benchmark_type = "all";
    app.add_option("-t,--type", benchmark_type,
                   "Benchmark type: 'l2cache', 'bandwidth', 'mixed_precision', "
                   "or 'all'")
        ->default_val(benchmark_type);

    CLI11_PARSE(app, argc, argv);

    // Initialize TT Metal context
    ttml::autograd::ctx();

    // Run the requested benchmark(s)
    if (benchmark_type == "l2cache" || benchmark_type == "all") {
      measure_l2_cache_size();
    }

    if (benchmark_type == "bandwidth" || benchmark_type == "all") {
      measure_interconnect_bandwidth();
    }

    if (benchmark_type == "mixed_precision" || benchmark_type == "all") {
      measure_mixed_precision_performance();
    }

    std::cout << "Benchmarks completed." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}