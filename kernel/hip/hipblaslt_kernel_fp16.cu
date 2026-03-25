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
constexpr int kAutoTuneHeuristicCount = 8;

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

void mm_hipblaslt_fp16_colmajor(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const std::optional<torch::stable::Tensor>& scale,
    const std::optional<torch::stable::Tensor>& bias,
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

    if(scale.has_value())
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

    if(bias.has_value())
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
    const float* const scale_ptr = scale.has_value() ? reinterpret_cast<const float*>(scale->const_data_ptr()) : nullptr;
    const half* const bias_ptr = bias.has_value() ? reinterpret_cast<const half*>(bias->const_data_ptr()) : nullptr;

    STD_TORCH_CHECK(is_aligned_16(a_ptr), "a data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(b_ptr), "b data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(d_ptr), "d data pointer must be 16-byte aligned");
    if(scale.has_value())
    {
        STD_TORCH_CHECK(is_aligned_16(scale_ptr), "scale data pointer must be 16-byte aligned");
    }
    if(bias.has_value())
    {
        STD_TORCH_CHECK(is_aligned_16(bias_ptr), "bias data pointer must be 16-byte aligned");
    }

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    auto& ctx = get_device_context(device_index);
    ensure_c_buffer(ctx, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(half));

    const float beta = 0.0f;
    const float alpha = static_cast<float>(alpha_scalar);

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
    epilogue.setMode(get_epilogue(bias.has_value(), use_relu));
    if(bias.has_value())
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
    if(scale.has_value())
    {
        inputs.setScaleAlphaVec(scale_ptr);
    }
    if(bias.has_value())
    {
        inputs.setBias(bias_ptr);
    }

    check_hipblaslt(gemm.setProblem(M, N, K, 1, epilogue, inputs), "hipblaslt_ext::Gemm::setProblem(colmajor)");

    const AlgoCacheKey cache_key{
        device_index,
        M,
        N,
        K,
        scale.has_value(),
        bias.has_value(),
        use_relu,
        1,
        static_cast<int>(solution_index),
    };

    hipblasStatus_t last_status = HIPBLAS_STATUS_NOT_SUPPORTED;

    const auto initialize_if_supported = [&](hipblasLtMatmulAlgo_t& algo) {
        size_t workspace_size = 0;
        last_status = gemm.isAlgoSupported(algo, workspace_size);
        if(last_status != HIPBLAS_STATUS_SUCCESS)
        {
            return false;
        }
        if(workspace_size > ctx.workspace_bytes)
        {
            last_status = HIPBLAS_STATUS_INVALID_VALUE;
            return false;
        }

        last_status = gemm.initialize(algo, ctx.workspace, true, stream);
        return last_status == HIPBLAS_STATUS_SUCCESS;
    };

    {
        std::lock_guard<std::mutex> lock(g_context_mutex);
        const auto it = g_algo_cache.find(cache_key);
        if(it != g_algo_cache.end())
        {
            auto cached_algo = it->second;
            if(initialize_if_supported(cached_algo))
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

    if(solution_index >= 0 || solution_index == -2)
    {
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
                            "hipblaslt_ext::getAllAlgos(colmajor)");

            for(const auto& candidate : all_algos)
            {
                auto algo = candidate.algo;
                if(hipblaslt_ext::getIndexFromAlgo(algo) != static_cast<int>(solution_index))
                {
                    continue;
                }

                if(initialize_if_supported(algo))
                {
                    std::lock_guard<std::mutex> lock(g_context_mutex);
                    g_algo_cache[cache_key] = algo;
                    initialized = true;
                    break;
                }
            }
        }
        else // solution_index == -2 (auto-tune)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_algos;
            check_hipblaslt(gemm.algoGetHeuristic(kAutoTuneHeuristicCount, pref, heuristic_algos),
                            "hipblaslt_ext::Gemm::algoGetHeuristic(colmajor autotune)");
            STD_TORCH_CHECK(!heuristic_algos.empty(),
                            "hipBLASLt returned no heuristic results for auto-tuning the column-major path");

            float best_ms = 1e9f;
            hipblasLtMatmulAlgo_t best_algo;
            bool found_best = false;

            hipEvent_t start, stop;
            check_hip(hipEventCreate(&start), "hipEventCreate");
            check_hip(hipEventCreate(&stop), "hipEventCreate");

            for(const auto& candidate : heuristic_algos)
            {
                auto algo = candidate.algo;
                if(initialize_if_supported(algo))
                {
                    // Warmup
                    bool warmup_failed = false;
                    for (int i = 0; i < 3; ++i) {
                        if (gemm.run(stream) != HIPBLAS_STATUS_SUCCESS) {
                            warmup_failed = true;
                            break;
                        }
                    }
                    
                    if (!warmup_failed) {
                        if (hipStreamSynchronize(stream) != hipSuccess) {
                            warmup_failed = true;
                        }
                    }
                    
                    if (warmup_failed) {
                        // Clear all pending errors
                        while (hipGetLastError() != hipSuccess) {}
                        continue;
                    }

                    check_hip(hipEventRecord(start, stream), "hipEventRecord");
                    const int iters = 10;
                    bool run_failed = false;
                    for (int i = 0; i < iters; ++i) {
                        if (gemm.run(stream) != HIPBLAS_STATUS_SUCCESS) {
                            run_failed = true;
                            break;
                        }
                    }
                    if (run_failed) {
                        continue;
                    }
                    check_hip(hipEventRecord(stop, stream), "hipEventRecord");

                    if (hipEventSynchronize(stop) != hipSuccess) {
                        (void)hipGetLastError(); // Clear any pending errors
                        continue;
                    }

                    float ms = 0;
                    check_hip(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime");

                    if (ms < best_ms) {
                        best_ms = ms;
                        best_algo = algo;
                        found_best = true;
                    }
                }
            }

            check_hip(hipEventDestroy(start), "hipEventDestroy");
            check_hip(hipEventDestroy(stop), "hipEventDestroy");

            if (found_best) {
                if(initialize_if_supported(best_algo)) {
                    std::lock_guard<std::mutex> lock(g_context_mutex);
                    g_algo_cache[cache_key] = best_algo;
                    initialized = true;
                    printf("hipBLASLt auto-tuning complete for M=%ld, N=%ld, K=%ld: best solution_index=%d (%.3f ms)\n", M, N, K, hipblaslt_ext::getIndexFromAlgo(best_algo), best_ms / 10.0f);
                }
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
            auto algo = heuristic.algo;
            if(initialize_if_supported(algo))
            {
                std::lock_guard<std::mutex> lock(g_context_mutex);
                g_algo_cache[cache_key] = algo;
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

} // namespace

STABLE_TORCH_LIBRARY(feather_ops, m)
{
    m.def(
        "mm_hipblaslt_fp16_colmajor("
        "Tensor a, "
        "Tensor b, "
        "Tensor? scale, "
        "Tensor? bias, "
        "Tensor(a!) d, "
        "float alpha_scalar, "
        "bool use_relu, "
        "int solution_index"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_ops, CUDA, m)
{
    m.impl("mm_hipblaslt_fp16_colmajor", TORCH_BOX(&mm_hipblaslt_fp16_colmajor));
}
