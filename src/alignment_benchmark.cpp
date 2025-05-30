// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include <algorithm>
#include <random>
#include <vector>
#include <cstdint>
#include <memory>

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
class Gather;
class ElementwiseAdd;
class SparseMV;
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
}  // namespace ttml

// Helper function to initialize arrays with random data
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

// Benchmark for aligned memory operations on Blackhole hardware
static void BM_BlackholeAlignedMemory(benchmark::State& state) {
    const size_t size = state.range(0);
    
    // Initialize host data
    std::vector<float> a_host(size), b_host(size), c_host(size);
    initialize_random_data(a_host, size);
    initialize_random_data(b_host, size);
    
    // Initialize TTML runtime
    auto runtime = ttml::Runtime::create();
    auto device = runtime->get_device(0); // Assuming Blackhole is the first device
    
    // Create a program and module
    auto program = ttml::Program::create(runtime);
    auto module = program->get_module();
    
    for (auto _ : state) {
        state.PauseTiming();
        
        // Create aligned tensors with different memory layouts
        auto a_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32, 
                                              ttml::Layout::create_aligned(64)); // 64-byte aligned
        auto b_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32, 
                                              ttml::Layout::create_aligned(64)); // 64-byte aligned
        auto c_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32, 
                                              ttml::Layout::create_aligned(64)); // 64-byte aligned
        
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

// Benchmark for unaligned memory operations on Blackhole hardware
static void BM_BlackholeUnalignedMemory(benchmark::State& state) {
    const size_t size = state.range(0);
    
    // Initialize host data
    std::vector<float> a_host(size), b_host(size), c_host(size);
    initialize_random_data(a_host, size);
    initialize_random_data(b_host, size);
    
    // Initialize TTML runtime
    auto runtime = ttml::Runtime::create();
    auto device = runtime->get_device(0); // Assuming Blackhole is the first device
    
    // Create a program and module
    auto program = ttml::Program::create(runtime);
    auto module = program->get_module();
    
    for (auto _ : state) {
        state.PauseTiming();
        
        // Create tensors with unaligned memory layout
        auto a_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32); // Default layout
        auto b_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32); // Default layout
        auto c_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32); // Default layout
        
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

// Benchmark for streaming memory operations (non-temporal) on Blackhole hardware
static void BM_BlackholeStreamingMemory(benchmark::State& state) {
    const size_t size = state.range(0);
    
    // Initialize host data
    std::vector<float> a_host(size), b_host(size), c_host(size);
    initialize_random_data(a_host, size);
    initialize_random_data(b_host, size);
    
    // Initialize TTML runtime
    auto runtime = ttml::Runtime::create();
    auto device = runtime->get_device(0); // Assuming Blackhole is the first device
    
    // Create a program and module
    auto program = ttml::Program::create(runtime);
    auto module = program->get_module();
    
    for (auto _ : state) {
        state.PauseTiming();
        
        // Create aligned tensors with streaming hint
        auto a_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32, 
                                              ttml::Layout::create_streaming()); // Streaming memory hint
        auto b_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32, 
                                              ttml::Layout::create_streaming()); // Streaming memory hint
        auto c_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32, 
                                              ttml::Layout::create_streaming()); // Streaming memory hint
        
        // Fill tensors with data
        a_tensor->fill_from(a_host.data());
        b_tensor->fill_from(b_host.data());
        
        state.ResumeTiming();
        
        // Create and execute an element-wise add operation optimized for streaming
        auto add_op = ttml::ElementwiseAdd::create(module, a_tensor, b_tensor, c_tensor,
                                                    ttml::OperationHint::STREAMING);
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

// Register benchmarks
BENCHMARK(BM_BlackholeAlignedMemory)->Range(1<<10, 1<<20);  // 1KB to 1MB
BENCHMARK(BM_BlackholeUnalignedMemory)->Range(1<<10, 1<<20);
BENCHMARK(BM_BlackholeStreamingMemory)->Range(1<<10, 1<<20);

BENCHMARK_MAIN(); 