#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t kWorkspaceBytes = 64ull * 1024ull * 1024ull;

struct DeviceContext {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    void* c_buffer = nullptr;
    size_t c_buffer_bytes = 0;
};

struct AlgoCacheKey {
    int device_index;
    int64_t m;
    int64_t n;
    int64_t k;
    bool has_scale;
    bool has_bias;
    bool use_relu;
    int layout_mode;
    int solution_index;

    bool operator==(const AlgoCacheKey& other) const = default;
};

struct AlgoCacheKeyHash {
    size_t operator()(const AlgoCacheKey& key) const
    {
        size_t h = static_cast<size_t>(key.device_index);
        h = h * 1315423911u + static_cast<size_t>(key.m);
        h = h * 1315423911u + static_cast<size_t>(key.n);
        h = h * 1315423911u + static_cast<size_t>(key.k);
        h = h * 1315423911u + static_cast<size_t>(key.has_scale);
        h = h * 1315423911u + static_cast<size_t>(key.has_bias);
        h = h * 1315423911u + static_cast<size_t>(key.use_relu);
        h = h * 1315423911u + static_cast<size_t>(key.layout_mode);
        h = h * 1315423911u + static_cast<size_t>(key.solution_index + 1);
        return h;
    }
};

std::mutex g_context_mutex;
std::unordered_map<int, DeviceContext> g_device_contexts;
std::unordered_map<AlgoCacheKey, hipblasLtMatmulAlgo_t, AlgoCacheKeyHash> g_algo_cache;

void check_hip(const hipError_t status, const char* const expr)
{
    STD_TORCH_CHECK(status == hipSuccess, expr, " failed: ", hipGetErrorString(status));
}

void check_hipblaslt(const hipblasStatus_t status, const char* const expr)
{
    STD_TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, expr, " failed with hipBLASLt status ", static_cast<int>(status));
}

DeviceContext& get_device_context(const int device_index)
{
    std::lock_guard<std::mutex> lock(g_context_mutex);
    auto& ctx = g_device_contexts[device_index];

    if(ctx.handle == nullptr)
    {
        check_hipblaslt(hipblasLtCreate(&ctx.handle), "hipblasLtCreate");
    }

    if(ctx.workspace_bytes < kWorkspaceBytes)
    {
        if(ctx.workspace != nullptr)
        {
            check_hip(hipFree(ctx.workspace), "hipFree(workspace)");
        }
        check_hip(hipMalloc(&ctx.workspace, kWorkspaceBytes), "hipMalloc(workspace)");
        ctx.workspace_bytes = kWorkspaceBytes;
    }

    return ctx;
}

void ensure_c_buffer(DeviceContext& ctx, const size_t bytes)
{
    if(ctx.c_buffer_bytes < bytes)
    {
        if(ctx.c_buffer != nullptr)
        {
            check_hip(hipFree(ctx.c_buffer), "hipFree(c_buffer)");
        }
        check_hip(hipMalloc(&ctx.c_buffer, bytes), "hipMalloc(c_buffer)");
        ctx.c_buffer_bytes = bytes;
    }
}

hipblasLtEpilogue_t get_epilogue(const bool has_bias, const bool use_relu)
{
    if(use_relu)
    {
        return has_bias ? HIPBLASLT_EPILOGUE_RELU_BIAS : HIPBLASLT_EPILOGUE_RELU;
    }
    return has_bias ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT;
}

void mm_hipblaslt_fp16(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const std::optional<torch::stable::Tensor>& scale,
    const std::optional<torch::stable::Tensor>& bias,
    const bool has_scale,
    const bool has_bias,
    torch::stable::Tensor& c,
    const double alpha_scalar,
    const bool use_relu)
{
    STD_TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    STD_TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    STD_TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");

    const auto device_index = a.get_device_index();
    STD_TORCH_CHECK(b.get_device_index() == device_index, "b must be on the same device as a");
    STD_TORCH_CHECK(c.get_device_index() == device_index, "c must be on the same device as a");

    STD_TORCH_CHECK(a.scalar_type() == torch::stable::ScalarType::Half, "a must be float16");
    STD_TORCH_CHECK(b.scalar_type() == torch::stable::ScalarType::Half, "b must be float16");
    STD_TORCH_CHECK(c.scalar_type() == torch::stable::ScalarType::Half, "c must be float16");

    STD_TORCH_CHECK(a.dim() == 2, "a must be 2D");
    STD_TORCH_CHECK(b.dim() == 2, "b must be 2D");
    STD_TORCH_CHECK(c.dim() == 2, "c must be 2D");

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = b.size(1);

    STD_TORCH_CHECK(b.size(0) == K, "b.shape[0] must equal a.shape[1]");
    STD_TORCH_CHECK(c.size(0) == M, "c.shape[0] must equal a.shape[0]");
    STD_TORCH_CHECK(c.size(1) == N, "c.shape[1] must equal b.shape[1]");

    // To preserve the fast path from the standalone benchmark, require row-major contiguous inputs.
    STD_TORCH_CHECK(a.stride(1) == 1, "a must have stride(1) == 1");
    STD_TORCH_CHECK(b.stride(1) == 1, "b must have stride(1) == 1");
    STD_TORCH_CHECK(c.stride(1) == 1, "c must have stride(1) == 1");
    STD_TORCH_CHECK(a.stride(0) == K, "a must be row-major contiguous");
    STD_TORCH_CHECK(b.stride(0) == N, "b must be row-major contiguous");
    STD_TORCH_CHECK(c.stride(0) == N, "c must be row-major contiguous");

    if(has_scale)
    {
        STD_TORCH_CHECK(scale.has_value(), "scale must be provided when has_scale=True");
        const auto& scale_t = *scale;
        STD_TORCH_CHECK(scale_t.is_cuda(), "scale must be a CUDA tensor");
        STD_TORCH_CHECK(scale_t.get_device_index() == device_index, "scale must be on the same device as a");
        STD_TORCH_CHECK(scale_t.scalar_type() == torch::stable::ScalarType::Float, "scale vector must be float32");
        STD_TORCH_CHECK(scale_t.dim() == 1, "scale vector must be 1D");
        STD_TORCH_CHECK(scale_t.numel() == N, "scale vector must have N elements");
        STD_TORCH_CHECK(scale_t.stride(0) == 1, "scale vector must be contiguous");
    }

    if(has_bias)
    {
        STD_TORCH_CHECK(bias.has_value(), "bias must be provided when has_bias=True");
        const auto& bias_t = *bias;
        STD_TORCH_CHECK(bias_t.is_cuda(), "bias must be a CUDA tensor");
        STD_TORCH_CHECK(bias_t.get_device_index() == device_index, "bias must be on the same device as a");
        STD_TORCH_CHECK(bias_t.scalar_type() == torch::stable::ScalarType::Half, "bias must be float16");
        STD_TORCH_CHECK(bias_t.dim() == 1, "bias must be 1D");
        STD_TORCH_CHECK(bias_t.numel() == N, "bias must have N elements");
        STD_TORCH_CHECK(bias_t.stride(0) == 1, "bias must be contiguous");
    }

    const auto is_aligned_16 = [](const void* const ptr) {
        return (reinterpret_cast<uintptr_t>(ptr) & 0xFu) == 0u;
    };

    const half* const a_ptr = reinterpret_cast<const half*>(a.const_data_ptr());
    const half* const b_ptr = reinterpret_cast<const half*>(b.const_data_ptr());
    half* const c_ptr = reinterpret_cast<half*>(c.mutable_data_ptr());
    const float* const scale_ptr = has_scale ? reinterpret_cast<const float*>(scale->const_data_ptr()) : nullptr;
    const half* const bias_ptr = has_bias ? reinterpret_cast<const half*>(bias->const_data_ptr()) : nullptr;

    STD_TORCH_CHECK(is_aligned_16(a_ptr), "a data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(b_ptr), "b data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(c_ptr), "c data pointer must be 16-byte aligned");
    if(has_scale)
    {
        STD_TORCH_CHECK(is_aligned_16(scale_ptr), "scale data pointer must be 16-byte aligned");
    }
    if(has_bias)
    {
        STD_TORCH_CHECK(is_aligned_16(bias_ptr), "bias data pointer must be 16-byte aligned");
    }

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    auto& ctx = get_device_context(device_index);
    ensure_c_buffer(ctx, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(half));

    // Map row-major A[M, K] @ B[K, N] to the benchmark-friendly column-major NN path:
    // D_col[N, M] = B_col[N, K] * A_col[K, M], stored into the same bytes as row-major C[M, N].
    const int64_t gemm_m = N;
    const int64_t gemm_n = M;
    const int64_t gemm_k = K;

    const int64_t lda = b.stride(0);
    const int64_t ldb = a.stride(0);
    const int64_t ldc = c.stride(0);
    const int64_t strideA = lda * b.size(0);
    const int64_t strideB = ldb * a.size(0);
    const int64_t strideC = ldc * c.size(0);

    float beta = 0.0f;
    float alpha = static_cast<float>(alpha_scalar);

    hipblaslt_ext::GemmProblemType problem_type(
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        HIP_R_16F,
        HIP_R_16F,
        HIP_R_16F,
        HIP_R_16F,
        HIPBLAS_COMPUTE_32F);
    hipblaslt_ext::Gemm gemm(
        ctx.handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        HIP_R_16F,
        HIP_R_16F,
        HIP_R_16F,
        HIP_R_16F,
        HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(ctx.workspace_bytes);
    gemm.setMaxWorkspaceBytes(ctx.workspace_bytes);

    hipblaslt_ext::GemmEpilogue epilogue;
    epilogue.setMode(get_epilogue(has_bias, use_relu));
    if(has_bias)
    {
        epilogue.setBiasDataType(HIP_R_16F);
    }

    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(b_ptr);
    inputs.setB(a_ptr);
    inputs.setC(ctx.c_buffer);
    inputs.setD(c_ptr);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    if(has_scale)
    {
        inputs.setScaleAlphaVec(scale_ptr);
    }
    if(has_bias)
    {
        inputs.setBias(bias_ptr);
    }

    check_hipblaslt(
        gemm.setProblem(
            gemm_m,
            gemm_n,
            gemm_k,
            1,
            lda,
            ldb,
            ldc,
            ldc,
            strideA,
            strideB,
            strideC,
            strideC,
            epilogue,
            inputs,
            problem_type),
        "hipblaslt_ext::Gemm::setProblem");

    const AlgoCacheKey cache_key{
        device_index,
        gemm_m,
        gemm_n,
        gemm_k,
        has_scale,
        has_bias,
        use_relu,
        0,
        -1,
    };

    {
        std::lock_guard<std::mutex> lock(g_context_mutex);
        auto it = g_algo_cache.find(cache_key);
        if(it != g_algo_cache.end())
        {
            if(gemm.initialize(it->second, ctx.workspace, true, stream) == HIPBLAS_STATUS_SUCCESS)
            {
                check_hipblaslt(gemm.run(stream), "hipblaslt_ext::Gemm::run");
                const hipError_t launch_err = hipGetLastError();
                STD_TORCH_CHECK(launch_err == hipSuccess, "hipBLASLt launch failed: ", hipGetErrorString(launch_err));
                return;
            }
            g_algo_cache.erase(it);
        }
    }

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
    check_hipblaslt(gemm.algoGetHeuristic(16, pref, heuristics), "hipblaslt_ext::Gemm::algoGetHeuristic");
    STD_TORCH_CHECK(!heuristics.empty(), "hipBLASLt returned no heuristic results");

    bool initialized = false;
    hipblasStatus_t last_status = HIPBLAS_STATUS_NOT_SUPPORTED;
    for(const auto& heuristic : heuristics)
    {
        last_status = gemm.initialize(heuristic.algo, ctx.workspace, true, stream);
        if(last_status == HIPBLAS_STATUS_SUCCESS)
        {
            std::lock_guard<std::mutex> lock(g_context_mutex);
            g_algo_cache[cache_key] = heuristic.algo;
            initialized = true;
            break;
        }
    }
    STD_TORCH_CHECK(initialized, "hipBLASLt failed to initialize any heuristic algorithm, last status ", static_cast<int>(last_status));

    check_hipblaslt(gemm.run(stream), "hipblaslt_ext::Gemm::run");

    const hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "hipBLASLt launch failed: ", hipGetErrorString(launch_err));
}

void mm_hipblaslt_fp16_colmajor(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const std::optional<torch::stable::Tensor>& scale,
    const std::optional<torch::stable::Tensor>& bias,
    const bool has_scale,
    const bool has_bias,
    torch::stable::Tensor& d,
    const double alpha_scalar,
    const bool use_relu,
    const int64_t solution_index)
{
    STD_TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    STD_TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    STD_TORCH_CHECK(d.is_cuda(), "d must be a CUDA tensor");

    const auto device_index = a.get_device_index();
    STD_TORCH_CHECK(b.get_device_index() == device_index, "b must be on the same device as a");
    STD_TORCH_CHECK(d.get_device_index() == device_index, "d must be on the same device as a");

    STD_TORCH_CHECK(a.scalar_type() == torch::stable::ScalarType::Half, "a must be float16");
    STD_TORCH_CHECK(b.scalar_type() == torch::stable::ScalarType::Half, "b must be float16");
    STD_TORCH_CHECK(d.scalar_type() == torch::stable::ScalarType::Half, "d must be float16");

    STD_TORCH_CHECK(a.dim() == 2, "a must be 2D");
    STD_TORCH_CHECK(b.dim() == 2, "b must be 2D");
    STD_TORCH_CHECK(d.dim() == 2, "d must be 2D");

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = b.size(1);

    STD_TORCH_CHECK(b.size(0) == K, "b.shape[0] must equal a.shape[1]");
    STD_TORCH_CHECK(d.size(0) == M, "d.shape[0] must equal a.shape[0]");
    STD_TORCH_CHECK(d.size(1) == N, "d.shape[1] must equal b.shape[1]");

    STD_TORCH_CHECK(a.stride(0) == 1, "a must be column-major contiguous (stride(0) == 1)");
    STD_TORCH_CHECK(b.stride(0) == 1, "b must be column-major contiguous (stride(0) == 1)");
    STD_TORCH_CHECK(d.stride(0) == 1, "d must be column-major contiguous (stride(0) == 1)");
    STD_TORCH_CHECK(a.stride(1) == M, "a must be column-major contiguous");
    STD_TORCH_CHECK(b.stride(1) == K, "b must be column-major contiguous");
    STD_TORCH_CHECK(d.stride(1) == M, "d must be column-major contiguous");

    if(has_scale)
    {
        STD_TORCH_CHECK(scale.has_value(), "scale must be provided when has_scale=True");
        const auto& scale_t = *scale;
        STD_TORCH_CHECK(scale_t.is_cuda(), "scale must be a CUDA tensor");
        STD_TORCH_CHECK(scale_t.get_device_index() == device_index, "scale must be on the same device as a");
        STD_TORCH_CHECK(scale_t.scalar_type() == torch::stable::ScalarType::Float, "scale vector must be float32");
        STD_TORCH_CHECK(scale_t.dim() == 1, "scale vector must be 1D");
        STD_TORCH_CHECK(scale_t.numel() == M, "scale vector must have M elements for the column-major fast path");
        STD_TORCH_CHECK(scale_t.stride(0) == 1, "scale vector must be contiguous");
    }

    if(has_bias)
    {
        STD_TORCH_CHECK(bias.has_value(), "bias must be provided when has_bias=True");
        const auto& bias_t = *bias;
        STD_TORCH_CHECK(bias_t.is_cuda(), "bias must be a CUDA tensor");
        STD_TORCH_CHECK(bias_t.get_device_index() == device_index, "bias must be on the same device as a");
        STD_TORCH_CHECK(bias_t.scalar_type() == torch::stable::ScalarType::Half, "bias must be float16");
        STD_TORCH_CHECK(bias_t.dim() == 1, "bias must be 1D");
        STD_TORCH_CHECK(bias_t.numel() == M, "bias must have M elements for the column-major fast path");
        STD_TORCH_CHECK(bias_t.stride(0) == 1, "bias must be contiguous");
    }

    const auto is_aligned_16 = [](const void* const ptr) {
        return (reinterpret_cast<uintptr_t>(ptr) & 0xFu) == 0u;
    };

    const half* const a_ptr = reinterpret_cast<const half*>(a.const_data_ptr());
    const half* const b_ptr = reinterpret_cast<const half*>(b.const_data_ptr());
    half* const d_ptr = reinterpret_cast<half*>(d.mutable_data_ptr());
    const float* const scale_ptr = has_scale ? reinterpret_cast<const float*>(scale->const_data_ptr()) : nullptr;
    const half* const bias_ptr = has_bias ? reinterpret_cast<const half*>(bias->const_data_ptr()) : nullptr;

    STD_TORCH_CHECK(is_aligned_16(a_ptr), "a data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(b_ptr), "b data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(d_ptr), "d data pointer must be 16-byte aligned");
    if(has_scale)
    {
        STD_TORCH_CHECK(is_aligned_16(scale_ptr), "scale data pointer must be 16-byte aligned");
    }
    if(has_bias)
    {
        STD_TORCH_CHECK(is_aligned_16(bias_ptr), "bias data pointer must be 16-byte aligned");
    }

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    auto& ctx = get_device_context(device_index);
    ensure_c_buffer(ctx, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(half));

    float beta = 0.0f;
    float alpha = static_cast<float>(alpha_scalar);

    hipblaslt_ext::Gemm gemm(
        ctx.handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        HIP_R_16F,
        HIP_R_16F,
        HIP_R_16F,
        HIP_R_16F,
        HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(ctx.workspace_bytes);
    gemm.setMaxWorkspaceBytes(ctx.workspace_bytes);

    hipblaslt_ext::GemmEpilogue epilogue;
    epilogue.setMode(get_epilogue(has_bias, use_relu));
    if(has_bias)
    {
        epilogue.setBiasDataType(HIP_R_16F);
    }

    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(a_ptr);
    inputs.setB(b_ptr);
    inputs.setC(ctx.c_buffer);
    inputs.setD(d_ptr);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    if(has_scale)
    {
        inputs.setScaleAlphaVec(scale_ptr);
    }
    if(has_bias)
    {
        inputs.setBias(bias_ptr);
    }

    check_hipblaslt(gemm.setProblem(M, N, K, 1, epilogue, inputs), "hipblaslt_ext::Gemm::setProblem(colmajor)");

    const AlgoCacheKey cache_key{
        device_index,
        M,
        N,
        K,
        has_scale,
        has_bias,
        use_relu,
        1,
        static_cast<int>(solution_index),
    };

    {
        std::lock_guard<std::mutex> lock(g_context_mutex);
        auto it = g_algo_cache.find(cache_key);
        if(it != g_algo_cache.end())
        {
            if(gemm.initialize(it->second, ctx.workspace, true, stream) == HIPBLAS_STATUS_SUCCESS)
            {
                check_hipblaslt(gemm.run(stream), "hipblaslt_ext::Gemm::run(colmajor)");
                const hipError_t launch_err = hipGetLastError();
                STD_TORCH_CHECK(launch_err == hipSuccess, "hipBLASLt launch failed: ", hipGetErrorString(launch_err));
                return;
            }
            g_algo_cache.erase(it);
        }
    }

    bool initialized = false;
    hipblasStatus_t last_status = HIPBLAS_STATUS_NOT_SUPPORTED;

    if(solution_index >= 0)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> all_algos;
        check_hipblaslt(hipblaslt_ext::getAllAlgos(ctx.handle,
                                                   hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                   HIPBLAS_OP_N,
                                                   HIPBLAS_OP_N,
                                                   HIP_R_16F,
                                                   HIP_R_16F,
                                                   HIP_R_16F,
                                                   HIP_R_16F,
                                                   HIPBLAS_COMPUTE_32F,
                                                   all_algos),
                        "hipblaslt_ext::getAllAlgos(colmajor)");

        for(const auto& candidate : all_algos)
        {
            auto algo = candidate.algo;
            if(hipblaslt_ext::getIndexFromAlgo(algo) != static_cast<int>(solution_index))
            {
                continue;
            }

            last_status = gemm.initialize(algo, ctx.workspace, true, stream);
            if(last_status == HIPBLAS_STATUS_SUCCESS)
            {
                std::lock_guard<std::mutex> lock(g_context_mutex);
                g_algo_cache[cache_key] = algo;
                initialized = true;
                break;
            }
        }
    }

    if(!initialized)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
        check_hipblaslt(gemm.algoGetHeuristic(16, pref, heuristics), "hipblaslt_ext::Gemm::algoGetHeuristic(colmajor)");
        STD_TORCH_CHECK(!heuristics.empty(), "hipBLASLt returned no heuristic results for column-major path");

        for(const auto& heuristic : heuristics)
        {
            last_status = gemm.initialize(heuristic.algo, ctx.workspace, true, stream);
            if(last_status == HIPBLAS_STATUS_SUCCESS)
            {
                std::lock_guard<std::mutex> lock(g_context_mutex);
                g_algo_cache[cache_key] = heuristic.algo;
                initialized = true;
                break;
            }
        }
    }

    STD_TORCH_CHECK(initialized,
                    "hipBLASLt failed to initialize any heuristic algorithm for the column-major path, last status ",
                    static_cast<int>(last_status));

    check_hipblaslt(gemm.run(stream), "hipblaslt_ext::Gemm::run(colmajor)");

    const hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "hipBLASLt launch failed: ", hipGetErrorString(launch_err));
}

double benchmark_raw_buffers(
    const torch::stable::Tensor& dummy,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t warmup_iters,
    const int64_t iters,
    const int64_t solution_index,
    const bool use_relu)
{
    const size_t a_elems = static_cast<size_t>(m) * k;
    const size_t b_elems = static_cast<size_t>(k) * n;
    const size_t c_d_elems = static_cast<size_t>(m) * n;
    const size_t bias_elems = static_cast<size_t>(m);
    const size_t scale_alpha_elems = static_cast<size_t>(m);

    const int device_index = 0; // Assuming device 0 for benchmark
    auto& ctx = get_device_context(device_index);

    half* d_a = nullptr;
    half* d_b = nullptr;
    half* d_c = nullptr;
    half* d_d = nullptr;
    half* d_bias = nullptr;
    float* d_scale = nullptr;

    check_hip(hipMalloc(&d_a, a_elems * sizeof(half)), "hipMalloc a");
    check_hip(hipMalloc(&d_b, b_elems * sizeof(half)), "hipMalloc b");
    check_hip(hipMalloc(&d_c, c_d_elems * sizeof(half)), "hipMalloc c");
    check_hip(hipMalloc(&d_d, c_d_elems * sizeof(half)), "hipMalloc d");
    check_hip(hipMalloc(&d_bias, bias_elems * sizeof(half)), "hipMalloc bias");
    check_hip(hipMalloc(&d_scale, scale_alpha_elems * sizeof(float)), "hipMalloc scale");

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    float alpha = 1.0f;
    float beta = 0.0f;

    hipblaslt_ext::Gemm gemm(
        ctx.handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        HIP_R_16F,
        HIP_R_16F,
        HIP_R_16F,
        HIP_R_16F,
        HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(ctx.workspace_bytes);
    gemm.setMaxWorkspaceBytes(ctx.workspace_bytes);

    hipblaslt_ext::GemmEpilogue epilogue;
    epilogue.setMode(get_epilogue(true, use_relu));
    epilogue.setBiasDataType(HIP_R_16F);

    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    inputs.setBias(d_bias);
    inputs.setScaleAlphaVec(d_scale);

    check_hipblaslt(gemm.setProblem(m, n, k, 1, epilogue, inputs), "hipblaslt_ext::Gemm::setProblem(raw)");

    bool initialized = false;
    hipblasStatus_t last_status = HIPBLAS_STATUS_NOT_SUPPORTED;

    if (solution_index >= 0)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> all_algos;
        check_hipblaslt(hipblaslt_ext::getAllAlgos(ctx.handle,
                                                   hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                   HIPBLAS_OP_N,
                                                   HIPBLAS_OP_N,
                                                   HIP_R_16F,
                                                   HIP_R_16F,
                                                   HIP_R_16F,
                                                   HIP_R_16F,
                                                   HIPBLAS_COMPUTE_32F,
                                                   all_algos),
                        "hipblaslt_ext::getAllAlgos(raw)");

        for(const auto& candidate : all_algos)
        {
            auto algo = candidate.algo;
            int idx = hipblaslt_ext::getIndexFromAlgo(algo);
            if(idx != static_cast<int>(solution_index))
            {
                continue;
            }

            last_status = gemm.initialize(algo, ctx.workspace, true, stream);
            if(last_status == HIPBLAS_STATUS_SUCCESS)
            {
                printf("Successfully initialized forced solution_index %d\n", idx);
                initialized = true;
                break;
            } else {
                printf("Failed to initialize forced solution_index %d, status = %d\n", idx, static_cast<int>(last_status));
            }
        }
    }

    if(!initialized)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
        check_hipblaslt(gemm.algoGetHeuristic(16, pref, heuristics), "algoGetHeuristic(raw)");
        STD_TORCH_CHECK(!heuristics.empty(), "no heuristic results");

        printf("algoGetHeuristic returned %zu results:\n", heuristics.size());
        for(size_t i = 0; i < heuristics.size(); ++i) {
            auto algo_copy = heuristics[i].algo;
            int idx = hipblaslt_ext::getIndexFromAlgo(algo_copy);
            printf("  [%zu] index=%d, workspace=%zu\n", i, idx, heuristics[i].workspaceSize);
        }

        for(const auto& heuristic : heuristics)
        {
            auto algo_copy = heuristic.algo;
            int idx = hipblaslt_ext::getIndexFromAlgo(algo_copy);
            last_status = gemm.initialize(algo_copy, ctx.workspace, true, stream);
            if(last_status == HIPBLAS_STATUS_SUCCESS)
            {
                printf("Successfully initialized heuristic solution_index %d\n", idx);
                initialized = true;
                break;
            } else {
                printf("Failed to initialize heuristic solution_index %d, status = %d\n", idx, static_cast<int>(last_status));
            }
        }
    }

    STD_TORCH_CHECK(initialized, "failed to initialize");

    for (int i = 0; i < warmup_iters; ++i)
    {
        check_hipblaslt(gemm.run(stream), "run warmup");
    }
    check_hip(hipStreamSynchronize(stream), "sync warmup");

    hipEvent_t start, stop;
    check_hip(hipEventCreate(&start), "create start");
    check_hip(hipEventCreate(&stop), "create stop");

    check_hip(hipEventRecord(start, stream), "record start");
    for (int i = 0; i < iters; ++i)
    {
        check_hipblaslt(gemm.run(stream), "run iters");
    }
    check_hip(hipEventRecord(stop, stream), "record stop");
    check_hip(hipEventSynchronize(stop), "sync stop");

    float elapsed_ms = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");

    check_hip(hipEventDestroy(start), "destroy start");
    check_hip(hipEventDestroy(stop), "destroy stop");
    check_hip(hipFree(d_a), "free a");
    check_hip(hipFree(d_b), "free b");
    check_hip(hipFree(d_c), "free c");
    check_hip(hipFree(d_d), "free d");
    check_hip(hipFree(d_bias), "free bias");
    check_hip(hipFree(d_scale), "free scale");

    return static_cast<double>(elapsed_ms) / static_cast<double>(iters);
}

} // namespace

STABLE_TORCH_LIBRARY(feather_ops, m)
{
    m.def(
        "mm_hipblaslt_fp16("
        "Tensor a, "
        "Tensor b, "
        "Tensor? scale, "
        "Tensor? bias, "
        "bool has_scale, "
        "bool has_bias, "
        "Tensor(a!) c, "
        "float alpha_scalar, "
        "bool use_relu"
        ") -> ()");
    m.def(
        "mm_hipblaslt_fp16_colmajor("
        "Tensor a, "
        "Tensor b, "
        "Tensor? scale, "
        "Tensor? bias, "
        "bool has_scale, "
        "bool has_bias, "
        "Tensor(a!) d, "
        "float alpha_scalar, "
        "bool use_relu, "
        "int solution_index"
        ") -> ()");
    m.def(
        "benchmark_raw_buffers("
        "Tensor dummy, "
        "int m, "
        "int n, "
        "int k, "
        "int warmup_iters, "
        "int iters, "
        "int solution_index, "
        "bool use_relu"
        ") -> float");
}

STABLE_TORCH_LIBRARY_IMPL(feather_ops, CUDA, m)
{
    m.impl("mm_hipblaslt_fp16", TORCH_BOX(&mm_hipblaslt_fp16));
    m.impl("mm_hipblaslt_fp16_colmajor", TORCH_BOX(&mm_hipblaslt_fp16_colmajor));
    m.impl("benchmark_raw_buffers", TORCH_BOX(&benchmark_raw_buffers));
}
