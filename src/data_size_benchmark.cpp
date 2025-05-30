// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

// Mock CLI11 header for completeness
namespace CLI {
class App {
public:
    static std::shared_ptr<App> create(const std::string& name = "") {
        return std::make_shared<App>();
    }

    template <typename T>
    void add_option(const std::string& name, T& value, const std::string& description = "") {
    }

    void parse(int argc, char** argv) {
    }
};
}  // namespace CLI

// Define the ttml namespace for benchmark purposes
namespace ttml {
// Forward declarations
class Tensor;
class Module;
class Program;
class Executable;
class Device;
class Runtime;
class ElementwiseAdd;
class MatMul;

class Layout {
public:
    static Layout create_aligned(int alignment) {
        return Layout();
    }
    static Layout create_streaming() {
        return Layout();
    }
};

enum class DataType { FLOAT32, INT32 };

enum class OperationHint { STREAMING };

class Tensor {
public:
    static std::shared_ptr<Tensor> create(std::vector<size_t> dims, DataType dtype, Layout layout = Layout()) {
        return std::make_shared<Tensor>();
    }

    void fill_from(const void* data) {
    }
    void copy_to(void* data) {
    }
};

class Module {
public:
    // Module methods
};

class Program {
public:
    static std::shared_ptr<Program> create(std::shared_ptr<Runtime> runtime) {
        return std::make_shared<Program>();
    }

    Module* get_module() {
        return &module_;
    }

    std::shared_ptr<Executable> compile() {
        return std::make_shared<Executable>();
    }

private:
    Module module_;
};

class Executable {
public:
    void run() {
    }
};

class Device {
public:
    void synchronize() {
    }
};

class Runtime {
public:
    static std::shared_ptr<Runtime> create() {
        return std::make_shared<Runtime>();
    }

    std::shared_ptr<Device> get_device(int index) {
        return std::make_shared<Device>();
    }
};

class ElementwiseAdd {
public:
    static std::shared_ptr<ElementwiseAdd> create(
        Module* module,
        std::shared_ptr<Tensor> a,
        std::shared_ptr<Tensor> b,
        std::shared_ptr<Tensor> c,
        OperationHint hint = OperationHint::STREAMING) {
        return std::make_shared<ElementwiseAdd>();
    }
};

class MatMul {
public:
    static std::shared_ptr<MatMul> create(
        Module* module, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, std::shared_ptr<Tensor> c) {
        return std::make_shared<MatMul>();
    }
};
}  // namespace ttml

// Helper function to initialize tensors with random data
template <typename T>
void initialize_random_data(std::vector<T>& data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(dist(gen));
    }
}

// Benchmark for vector addition at different data sizes using Blackhole hardware
static void BM_BlackholeVectorAdd(benchmark::State& state) {
    const size_t size = state.range(0);

    // Initialize host data
    std::vector<float> a_host(size), b_host(size), c_host(size);
    initialize_random_data(a_host, size);
    initialize_random_data(b_host, size);

    // Initialize TTML runtime
    auto runtime = ttml::Runtime::create();
    auto device = runtime->get_device(0);  // Assuming Blackhole is the first device

    // Create a program and module
    auto program = ttml::Program::create(runtime);
    auto module = program->get_module();

    for (auto _ : state) {
        state.PauseTiming();

        // Create tensors
        auto a_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32);
        auto b_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32);
        auto c_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32);

        // Fill tensors with data
        a_tensor->fill_from(a_host.data());
        b_tensor->fill_from(b_host.data());

        state.ResumeTiming();

        // Create and execute an element-wise add operation
        auto add_op = ttml::ElementwiseAdd::create(module, a_tensor, b_tensor, c_tensor);
        auto executable = program->compile();
        executable->run();

        // Synchronize to ensure operation completes
        device->synchronize();

        state.PauseTiming();

// Copy result back to host (if needed for verification)
#ifndef NDEBUG
        c_tensor->copy_to(c_host.data());
#endif

        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size) * int64_t(sizeof(float) * 3));
}

// Benchmark for matrix multiplication at different sizes using Blackhole hardware
static void BM_BlackholeMatrixMul(benchmark::State& state) {
    const size_t M = state.range(0);
    const size_t N = state.range(0);
    const size_t K = state.range(0);

    // Initialize host data
    std::vector<float> a_host(M * K), b_host(K * N), c_host(M * N);
    initialize_random_data(a_host, M * K);
    initialize_random_data(b_host, K * N);

    // Initialize TTML runtime
    auto runtime = ttml::Runtime::create();
    auto device = runtime->get_device(0);  // Assuming Blackhole is the first device

    // Create a program and module
    auto program = ttml::Program::create(runtime);
    auto module = program->get_module();

    for (auto _ : state) {
        state.PauseTiming();

        // Create tensors
        auto a_tensor = ttml::Tensor::create({M, K}, ttml::DataType::FLOAT32);
        auto b_tensor = ttml::Tensor::create({K, N}, ttml::DataType::FLOAT32);
        auto c_tensor = ttml::Tensor::create({M, N}, ttml::DataType::FLOAT32);

        // Fill tensors with data
        a_tensor->fill_from(a_host.data());
        b_tensor->fill_from(b_host.data());

        state.ResumeTiming();

        // Create and execute a matrix multiplication operation
        auto matmul_op = ttml::MatMul::create(module, a_tensor, b_tensor, c_tensor);
        auto executable = program->compile();
        executable->run();

        // Synchronize to ensure operation completes
        device->synchronize();

        state.PauseTiming();

// Copy result back to host (if needed for verification)
#ifndef NDEBUG
        c_tensor->copy_to(c_host.data());
#endif

        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(M * N * K) * int64_t(sizeof(float) * 2));
    state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(M * N * K) * 2);
}

// Register benchmarks with different data sizes
BENCHMARK(BM_BlackholeVectorAdd)->Range(8, 8 << 13);  // 8 to 65536
BENCHMARK(BM_BlackholeMatrixMul)->Range(8, 1024);     // Matrix sizes from 8x8 to 1024x1024

BENCHMARK_MAIN();