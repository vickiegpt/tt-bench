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
#include <numeric>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"

using namespace std::chrono;

// Constants
constexpr uint32_t WARMUP_ITERATIONS = 5;
constexpr uint32_t BENCHMARK_ITERATIONS = 10;
constexpr uint32_t MIN_MATRIX_SIZE = 32;    // Minimum matrix size to test
constexpr uint32_t MAX_MATRIX_SIZE = 4096;  // Maximum matrix size to test

// Helper function to get time in microseconds
uint64_t get_time_us() {
    return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

// Function to measure L2 cache size through tensor operations
void measure_l2_cache_size() {
    std::cout << "\n===== Measuring Tensix L2 Cache Size through Tensor Operations =====" << std::endl;

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

        auto a_tensor_data = ttml::core::from_vector(a_data, shape, device);
        auto b_tensor_data = ttml::core::from_vector(b_data, shape, device);

        auto a_tensor = ttml::autograd::create_tensor(a_tensor_data);
        auto b_tensor = ttml::autograd::create_tensor(b_tensor_data);

        // Warmup runs
        for (uint32_t i = 0; i < WARMUP_ITERATIONS; i++) {
            auto result = ttml::ops::mul(a_tensor, b_tensor);
            result->get_value();  // Force computation
        }

        // Benchmark runs
        std::vector<double> times;
        for (uint32_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
            auto start_time = get_time_us();
            auto result = ttml::ops::mul(a_tensor, b_tensor);
            result->get_value();  // Force computation
            auto end_time = get_time_us();

            double elapsed_us = static_cast<double>(end_time - start_time);
            double us_per_element = elapsed_us / (size * size);

            times.push_back(us_per_element);
        }

        // Calculate average latency
        double avg_latency = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        uint32_t tensor_size_kb = (size * size * sizeof(float)) / 1024;  // Estimate memory footprint
        latencies.push_back({tensor_size_kb, avg_latency});

        std::cout << "  Average latency: " << avg_latency << " us per element" << std::endl;
        std::cout << "  Estimated memory footprint: " << tensor_size_kb << " KB" << std::endl;
    }

    // Analyze results to determine L2 cache size
    uint32_t l2_cache_size_kb = 0;
    double max_latency_increase = 0.0;

    // Look for the point where latency significantly increases (cache miss)
    for (size_t i = 1; i < latencies.size(); i++) {
        double latency_increase = latencies[i].second / latencies[i - 1].second;

        std::cout << "Size transition " << latencies[i - 1].first << "KB -> " << latencies[i].first
                  << "KB: Latency increase factor: " << latency_increase << std::endl;

        if (latency_increase > max_latency_increase) {
            max_latency_increase = latency_increase;
            l2_cache_size_kb = latencies[i - 1].first;
        }
    }

    std::cout << "\nEstimated L2 cache size: " << l2_cache_size_kb << " KB" << std::endl;

    // Write results to file
    std::ofstream result_file("l2_cache_results.csv");
    result_file << "Size_KB,Latency_us_Per_Element\n";
    for (const auto& [size, latency] : latencies) {
        result_file << size << "," << latency << "\n";
    }
    result_file.close();
}

// Function to measure interconnect bandwidth through tensor data movement
void measure_interconnect_bandwidth() {
    std::cout << "\n===== Measuring Interconnect Bandwidth through Tensor Transfers =====" << std::endl;

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
            auto tensor_data = ttml::core::from_vector(data, shape, device);
            auto tensor = ttml::autograd::create_tensor(tensor_data);

            // Force a copy operation (data movement)
            auto tensor_copy = ttml::autograd::create_tensor(tensor->get_value());

            auto end_time = get_time_us();

            double elapsed_us = static_cast<double>(end_time - start_time);
            double bytes_transferred = size * size * sizeof(float) * 2;  // Count both to and from device
            double bandwidth_mb_per_s = (bytes_transferred / 1024.0 / 1024.0) / (elapsed_us / 1000000.0);

            bandwidths.push_back(bandwidth_mb_per_s);
            std::cout << "  Run " << i << ": " << bandwidth_mb_per_s << " MB/s" << std::endl;
        }

        // Calculate average bandwidth
        double avg_bandwidth = std::accumulate(bandwidths.begin(), bandwidths.end(), 0.0) / bandwidths.size();
        bandwidth_results.push_back({data_size_kb, avg_bandwidth});

        std::cout << "  Average bandwidth: " << avg_bandwidth << " MB/s" << std::endl;
    }

    // Write results to file
    std::ofstream result_file("bandwidth_results.csv");
    result_file << "Size_KB,Bandwidth_MB_per_s\n";
    for (const auto& [size, bw] : bandwidth_results) {
        result_file << size << "," << bw << "\n";
    }
    result_file.close();

    // Find maximum bandwidth
    auto max_bw =
        std::max_element(bandwidth_results.begin(), bandwidth_results.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

    std::cout << "\nMaximum interconnect bandwidth: " << max_bw->second << " MB/s at " << max_bw->first << " KB"
              << std::endl;
}

int main(int argc, char** argv) {
    try {
        // Parse command line arguments
        CLI::App app{"Tensix Hardware Benchmark"};
        std::string benchmark_type = "all";
        app.add_option("-t,--type", benchmark_type, "Benchmark type: 'l2cache', 'bandwidth', or 'all'")
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

        std::cout << "Benchmarks completed." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}