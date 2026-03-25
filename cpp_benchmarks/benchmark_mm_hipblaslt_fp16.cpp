#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#define CHECK_HIP(cmd)                                                                                        \
    do                                                                                                        \
    {                                                                                                         \
        hipError_t error__ = (cmd);                                                                           \
        if (error__ != hipSuccess)                                                                            \
        {                                                                                                     \
            std::cerr << "HIP error: " << hipGetErrorString(error__) << " at " << __FILE__ << ':' << __LINE__ \
                      << std::endl;                                                                           \
            std::exit(EXIT_FAILURE);                                                                          \
        }                                                                                                     \
    } while (false)

#define CHECK_HIPBLASLT(cmd)                                                                                        \
    do                                                                                                              \
    {                                                                                                               \
        hipblasStatus_t status__ = (cmd);                                                                           \
        if (status__ != HIPBLAS_STATUS_SUCCESS)                                                                     \
        {                                                                                                           \
            std::cerr << "hipBLASLt error: " << static_cast<int>(status__) << " at " << __FILE__ << ':' << __LINE__ \
                      << std::endl;                                                                                 \
            std::exit(EXIT_FAILURE);                                                                                \
        }                                                                                                           \
    } while (false)

struct Options
{
    int64_t m = 8192;
    int64_t n = 8192;
    int64_t k = 8192;
    int64_t batch_count = 1;
    int warmup_iters = 10;
    int iters = 100;
    size_t workspace_mb = 64;
};

void print_usage(const char* argv0)
{
    std::cout << "Usage: " << argv0
              << " [--m M] [--n N] [--k K] [--batch B] [--warmup W] [--iters I] [--workspace-mb MB]\n"
              << "Defaults: --m 8192 --n 8192 --k 8192 --batch 1 --warmup 3 --iters 10 --workspace-mb 64\n";
}

Options parse_args(int argc, char** argv)
{
    Options opts;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto need_value = [&](const char* name)
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(EXIT_FAILURE);
            }
        };

        if (arg == "--m")
        {
            need_value("--m");
            opts.m = std::stoll(argv[++i]);
        }
        else if (arg == "--n")
        {
            need_value("--n");
            opts.n = std::stoll(argv[++i]);
        }
        else if (arg == "--k")
        {
            need_value("--k");
            opts.k = std::stoll(argv[++i]);
        }
        else if (arg == "--batch")
        {
            need_value("--batch");
            opts.batch_count = std::stoll(argv[++i]);
        }
        else if (arg == "--warmup")
        {
            need_value("--warmup");
            opts.warmup_iters = std::stoi(argv[++i]);
        }
        else if (arg == "--iters")
        {
            need_value("--iters");
            opts.iters = std::stoi(argv[++i]);
        }
        else if (arg == "--workspace-mb")
        {
            need_value("--workspace-mb");
            opts.workspace_mb = static_cast<size_t>(std::stoull(argv[++i]));
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (opts.m <= 0 || opts.n <= 0 || opts.k <= 0 || opts.batch_count <= 0 || opts.warmup_iters < 0 || opts.iters <= 0)
    {
        std::cerr << "All sizes must be positive, warmup must be non-negative, and iters must be positive."
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return opts;
}

template <typename T>
T* device_alloc(size_t count)
{
    T* ptr = nullptr;
    CHECK_HIP(hipMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

void fill_random_half(half* ptr, size_t size) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<half> host_data(size);
    for (size_t i = 0; i < size; ++i) {
        host_data[i] = static_cast<half>(dist(gen));
    }
    CHECK_HIP(hipMemcpy(ptr, host_data.data(), size * sizeof(half), hipMemcpyHostToDevice));
}

void fill_random_float(float* ptr, size_t size) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> host_data(size);
    for (size_t i = 0; i < size; ++i) {
        host_data[i] = dist(gen);
    }
    CHECK_HIP(hipMemcpy(ptr, host_data.data(), size * sizeof(float), hipMemcpyHostToDevice));
}

int main(int argc, char** argv)
{
    const Options opts = parse_args(argc, argv);

    const size_t a_elems = static_cast<size_t>(opts.m) * opts.k * opts.batch_count;
    const size_t b_elems = static_cast<size_t>(opts.k) * opts.n * opts.batch_count;
    const size_t c_d_elems = static_cast<size_t>(opts.m) * opts.n * opts.batch_count;
    const size_t bias_elems = static_cast<size_t>(opts.m);
    const size_t scale_alpha_elems = static_cast<size_t>(opts.m);
    const size_t workspace_bytes = opts.workspace_mb * 1024ULL * 1024ULL;
    const double gemm_flops = 2.0 * static_cast<double>(opts.m) * opts.n * opts.k * opts.batch_count;
    const double matrix_bytes_total = static_cast<double>(a_elems + b_elems + 2 * c_d_elems) * sizeof(half) +
                                      static_cast<double>(bias_elems) * sizeof(half) +
                                      static_cast<double>(scale_alpha_elems) * sizeof(float) +
                                      static_cast<double>(workspace_bytes);

    hipblasLtHandle_t handle = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    CHECK_HIPBLASLT(hipblasLtCreate(&handle));
    CHECK_HIP(hipStreamCreate(&stream));
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    half* d_a = device_alloc<half>(a_elems);
    half* d_b = device_alloc<half>(b_elems);
    half* d_c = device_alloc<half>(c_d_elems);
    half* d_d = device_alloc<half>(c_d_elems);
    half* d_bias = device_alloc<half>(bias_elems);
    float* d_scale_alpha_vec = device_alloc<float>(scale_alpha_elems);
    void* d_workspace = nullptr;

    if (workspace_bytes > 0)
    {
        CHECK_HIP(hipMalloc(&d_workspace, workspace_bytes));
    }

    fill_random_half(d_a, a_elems);
    fill_random_half(d_b, b_elems);
    CHECK_HIP(hipMemset(d_c, 0, c_d_elems * sizeof(half)));
    CHECK_HIP(hipMemset(d_d, 0, c_d_elems * sizeof(half)));
    fill_random_half(d_bias, bias_elems);
    fill_random_float(d_scale_alpha_vec, scale_alpha_elems);
    CHECK_HIP(hipStreamSynchronize(stream));

    float alpha = 1.0f;
    float beta = 0.0f;

    hipblaslt_ext::GemmPreference pref;
    pref.setMaxWorkspaceBytes(workspace_bytes);

    hipblaslt_ext::Gemm gemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F,
                             HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogue epilogue;
    epilogue.setMode(HIPBLASLT_EPILOGUE_RELU_BIAS);
    epilogue.setBiasDataType(HIP_R_16F);

    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(d_a);
    inputs.setB(d_b);
    inputs.setC(d_c);
    inputs.setD(d_d);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    inputs.setBias(d_bias);
    inputs.setScaleAlphaVec(d_scale_alpha_vec);

    CHECK_HIPBLASLT(gemm.setProblem(opts.m, opts.n, opts.k, opts.batch_count, epilogue, inputs));
    gemm.setMaxWorkspaceBytes(workspace_bytes);

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
    CHECK_HIPBLASLT(gemm.algoGetHeuristic(16, pref, heuristics));
    if (heuristics.empty())
    {
        std::cerr << "No hipBLASLt heuristic result found. Try increasing --workspace-mb or using a smaller size."
                  << std::endl;
        return EXIT_FAILURE;
    }

    size_t chosen_workspace = 0;
    size_t chosen_index = 0;
    hipblasStatus_t init_status = HIPBLAS_STATUS_NOT_SUPPORTED;
    bool found_algo = false;
    for (size_t i = 0; i < heuristics.size(); ++i)
    {
        init_status = gemm.initialize(heuristics[i].algo, d_workspace, true, stream);
        if (init_status == HIPBLAS_STATUS_SUCCESS)
        {
            chosen_workspace = heuristics[i].workspaceSize;
            chosen_index = i;
            found_algo = true;
            break;
        }
    }

    if (!found_algo)
    {
        std::cerr << "Found heuristic results, but none initialized successfully with user args. Last status: "
                  << static_cast<int>(init_status) << std::endl;
        return EXIT_FAILURE;
    }

    for (int i = 0; i < opts.warmup_iters; ++i)
    {
        CHECK_HIPBLASLT(gemm.run(stream));
    }
    CHECK_HIP(hipStreamSynchronize(stream));

    std::vector<float> timings_ms;
    timings_ms.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i)
    {
        CHECK_HIP(hipEventRecord(start, stream));
        CHECK_HIPBLASLT(gemm.run(stream));
        CHECK_HIP(hipEventRecord(stop, stream));
        CHECK_HIP(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
        timings_ms.push_back(elapsed_ms);
    }

    const float total_ms = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0f);
    const float avg_ms = total_ms / static_cast<float>(timings_ms.size());
    const float min_ms = *std::min_element(timings_ms.begin(), timings_ms.end());
    const float max_ms = *std::max_element(timings_ms.begin(), timings_ms.end());

    const double avg_tflops = gemm_flops / static_cast<double>(avg_ms) / 1.0e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "hipBLASLt FP16 benchmark\n";
    std::cout << "  problem: m=" << opts.m << " n=" << opts.n << " k=" << opts.k << " batch=" << opts.batch_count
              << "\n";
    std::cout << "  mode: NN, fp16 inputs/outputs, fp32 compute, relu+bias, scaleAlphaVec, user args\n";
    std::cout << "  heuristic index: " << chosen_index << '\n';
    std::cout << "  workspace: " << (chosen_workspace / (1024.0 * 1024.0)) << " MiB used, " << opts.workspace_mb
              << " MiB reserved\n";
    std::cout << "  memory footprint: " << (matrix_bytes_total / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "  solution: " << gemm.getSolutionName() << '\n';
    std::cout << "  kernel: " << gemm.getKernelName() << '\n';
    std::cout << "  avg ms: " << avg_ms << "\n";
    std::cout << "  min ms: " << min_ms << "\n";
    std::cout << "  max ms: " << max_ms << "\n";
    std::cout << "  avg TFLOP/s: " << avg_tflops << "\n";

    CHECK_HIP(hipFree(d_workspace));
    CHECK_HIP(hipFree(d_scale_alpha_vec));
    CHECK_HIP(hipFree(d_bias));
    CHECK_HIP(hipFree(d_d));
    CHECK_HIP(hipFree(d_c));
    CHECK_HIP(hipFree(d_b));
    CHECK_HIP(hipFree(d_a));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIPBLASLT(hipblasLtDestroy(handle));

    return EXIT_SUCCESS;
}
