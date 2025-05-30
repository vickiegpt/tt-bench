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

class Gather {
public:
    static std::shared_ptr<Gather> create(
        Module* module, std::shared_ptr<Tensor> src, std::shared_ptr<Tensor> indices, std::shared_ptr<Tensor> dst) {
        return std::make_shared<Gather>();
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

class SparseMV {
public:
    static std::shared_ptr<SparseMV> create(
        Module* module,
        std::shared_ptr<Tensor> values,
        std::shared_ptr<Tensor> col_indices,
        std::shared_ptr<Tensor> row_ptrs,
        std::shared_ptr<Tensor> x,
        std::shared_ptr<Tensor> y) {
        return std::make_shared<SparseMV>();
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

// Helper function to initialize array with random data
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

// Helper function to generate random indices for sparse operations
void generate_random_indices(std::vector<int32_t>& indices, size_t size, size_t max_index) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(0, max_index - 1);

    indices.resize(size);
    for (size_t i = 0; i < size; ++i) {
        indices[i] = dist(gen);
    }
}

// Helper function to generate clustered indices (to test locality)
void generate_clustered_indices(std::vector<int32_t>& indices, size_t size, size_t max_index, size_t cluster_size) {
    std::random_device rd;
    std::mt19937 gen(rd());

    indices.resize(size);

    // Generate clusters
    size_t num_clusters = (size + cluster_size - 1) / cluster_size;
    std::uniform_int_distribution<int32_t> cluster_dist(0, max_index / cluster_size - 1);

    for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
        int32_t base_idx = cluster_dist(gen) * cluster_size;
        size_t end = std::min((cluster + 1) * cluster_size, size);

        for (size_t i = cluster * cluster_size; i < end; ++i) {
            indices[i] = base_idx + (i - cluster * cluster_size);
        }
    }

    // Shuffle within clusters for some randomness
    for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
        size_t start = cluster * cluster_size;
        size_t end = std::min((cluster + 1) * cluster_size, size);
        std::shuffle(indices.begin() + start, indices.begin() + end, gen);
    }
}

// Benchmark for gather operations with random indices on Blackhole
static void BM_BlackholeGatherRandom(benchmark::State& state) {
    const size_t size = state.range(0);
    const size_t data_size = state.range(1);

    // Initialize host data
    std::vector<float> src_host(data_size);
    std::vector<float> dst_host(size);
    std::vector<int32_t> indices_host;

    initialize_random_data(src_host, data_size);
    generate_random_indices(indices_host, size, data_size);

    // Initialize TTML runtime
    auto runtime = ttml::Runtime::create();
    auto device = runtime->get_device(0);  // Assuming Blackhole is the first device

    // Create a program and module
    auto program = ttml::Program::create(runtime);
    auto module = program->get_module();

    for (auto _ : state) {
        state.PauseTiming();

        // Create tensors
        auto src_tensor = ttml::Tensor::create({data_size}, ttml::DataType::FLOAT32);
        auto indices_tensor = ttml::Tensor::create({size}, ttml::DataType::INT32);
        auto dst_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32);

        // Fill tensors with data
        src_tensor->fill_from(src_host.data());
        indices_tensor->fill_from(indices_host.data());

        state.ResumeTiming();

        // Create and execute a gather operation
        auto gather_op = ttml::Gather::create(module, src_tensor, indices_tensor, dst_tensor);
        auto executable = program->compile();
        executable->run();

        // Synchronize to ensure operation completes
        device->synchronize();

        state.PauseTiming();

// Copy result back to host (if needed for verification)
#ifndef NDEBUG
        dst_tensor->copy_to(dst_host.data());
#endif

        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size) * int64_t(sizeof(float)));
}

// Benchmark for gather operations with clustered indices on Blackhole
static void BM_BlackholeGatherClustered(benchmark::State& state) {
    const size_t size = state.range(0);
    const size_t data_size = state.range(1);
    const size_t cluster_size = 16;  // Fixed cluster size

    // Initialize host data
    std::vector<float> src_host(data_size);
    std::vector<float> dst_host(size);
    std::vector<int32_t> indices_host;

    initialize_random_data(src_host, data_size);
    generate_clustered_indices(indices_host, size, data_size, cluster_size);

    // Initialize TTML runtime
    auto runtime = ttml::Runtime::create();
    auto device = runtime->get_device(0);  // Assuming Blackhole is the first device

    // Create a program and module
    auto program = ttml::Program::create(runtime);
    auto module = program->get_module();

    for (auto _ : state) {
        state.PauseTiming();

        // Create tensors
        auto src_tensor = ttml::Tensor::create({data_size}, ttml::DataType::FLOAT32);
        auto indices_tensor = ttml::Tensor::create({size}, ttml::DataType::INT32);
        auto dst_tensor = ttml::Tensor::create({size}, ttml::DataType::FLOAT32);

        // Fill tensors with data
        src_tensor->fill_from(src_host.data());
        indices_tensor->fill_from(indices_host.data());

        state.ResumeTiming();

        // Create and execute a gather operation
        auto gather_op = ttml::Gather::create(module, src_tensor, indices_tensor, dst_tensor);
        auto executable = program->compile();
        executable->run();

        // Synchronize to ensure operation completes
        device->synchronize();

        state.PauseTiming();

// Copy result back to host (if needed for verification)
#ifndef NDEBUG
        dst_tensor->copy_to(dst_host.data());
#endif

        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size) * int64_t(sizeof(float)));
}

// Benchmark for sparse matrix-vector multiplication on Blackhole
static void BM_BlackholeSparseMV(benchmark::State& state) {
    const int num_rows = state.range(0);
    const int num_cols = state.range(0);  // Square matrix
    const float density = 0.01f;          // 1% non-zero elements

    // Calculate number of non-zero elements
    const int nnz = static_cast<int>(num_rows * num_cols * density);

    // Initialize host data for CSR format
    std::vector<float> values(nnz);
    std::vector<int32_t> col_indices(nnz);
    std::vector<int32_t> row_ptrs(num_rows + 1, 0);
    std::vector<float> x_host(num_cols);
    std::vector<float> y_host(num_rows);

    // Generate random sparse matrix (simplified for benchmark)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> val_dist(-10.0f, 10.0f);
    std::uniform_int_distribution<int32_t> col_dist(0, num_cols - 1);

    initialize_random_data(x_host, num_cols);

    // Simplified CSR generation - distributing nnz values evenly
    int avg_nnz_per_row = nnz / num_rows;
    int curr_nnz = 0;

    for (int row = 0; row < num_rows; ++row) {
        row_ptrs[row] = curr_nnz;

        for (int i = 0; i < avg_nnz_per_row && curr_nnz < nnz; ++i) {
            col_indices[curr_nnz] = col_dist(gen);
            values[curr_nnz] = val_dist(gen);
            curr_nnz++;
        }
    }
    row_ptrs[num_rows] = curr_nnz;

    // Initialize TTML runtime
    auto runtime = ttml::Runtime::create();
    auto device = runtime->get_device(0);  // Assuming Blackhole is the first device

    // Create a program and module
    auto program = ttml::Program::create(runtime);
    auto module = program->get_module();

    for (auto _ : state) {
        state.PauseTiming();

        // Create tensors
        auto values_tensor = ttml::Tensor::create({static_cast<size_t>(nnz)}, ttml::DataType::FLOAT32);
        auto col_indices_tensor = ttml::Tensor::create({static_cast<size_t>(nnz)}, ttml::DataType::INT32);
        auto row_ptrs_tensor = ttml::Tensor::create({static_cast<size_t>(num_rows + 1)}, ttml::DataType::INT32);
        auto x_tensor = ttml::Tensor::create({static_cast<size_t>(num_cols)}, ttml::DataType::FLOAT32);
        auto y_tensor = ttml::Tensor::create({static_cast<size_t>(num_rows)}, ttml::DataType::FLOAT32);

        // Fill tensors with data
        values_tensor->fill_from(values.data());
        col_indices_tensor->fill_from(col_indices.data());
        row_ptrs_tensor->fill_from(row_ptrs.data());
        x_tensor->fill_from(x_host.data());

        state.ResumeTiming();

        // Create and execute a sparse matrix-vector multiplication operation
        auto spmv_op =
            ttml::SparseMV::create(module, values_tensor, col_indices_tensor, row_ptrs_tensor, x_tensor, y_tensor);
        auto executable = program->compile();
        executable->run();

        // Synchronize to ensure operation completes
        device->synchronize();

        state.PauseTiming();

// Copy result back to host (if needed for verification)
#ifndef NDEBUG
        y_tensor->copy_to(y_host.data());
#endif

        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(nnz) * int64_t(sizeof(float) * 2));
}

// Register benchmarks
BENCHMARK(BM_BlackholeGatherRandom)->Ranges({{1 << 10, 1 << 18}, {1 << 10, 1 << 20}});
BENCHMARK(BM_BlackholeGatherClustered)->Ranges({{1 << 10, 1 << 18}, {1 << 10, 1 << 20}});
BENCHMARK(BM_BlackholeSparseMV)->Range(1 << 8, 1 << 12);  // 256 to 4096 rows/cols

BENCHMARK_MAIN();