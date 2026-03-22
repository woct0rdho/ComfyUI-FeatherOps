#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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

extern "C" bool launch_scaled_mm(
    const half* a,
    const uint8_t* b_prepacked,
    const half* scale,
    const half* bias,
    half* c,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t stride_am,
    const int64_t stride_cm,
    const int has_scale,
    const int has_bias,
    const int block_warps_m,
    const int block_warps_n,
    const int unroll_k,
    const int repeat_m,
    const int repeat_n,
    const int b_dtype,
    hipStream_t stream);

struct Options
{
    int64_t m = 8192;
    int64_t n = 8192;
    int64_t k = 8192;
    int warmup_iters = 10;
    int iters = 100;
    int block_warps_m = 1;
    int block_warps_n = 8;
    int unroll_k = 4;
    int repeat_m = 8;
    int repeat_n = 2;
};

void print_usage(const char* argv0)
{
    std::cout << "Usage: " << argv0
              << " [--m M] [--n N] [--k K] [--warmup W] [--iters I] [--warps_m M] [--warps_n N] [--unroll U] [--repeat_m RM] [--repeat_n RN]\n"
              << "Defaults: --m 8192 --n 8192 --k 8192 --warmup 10 --iters 50 --warps_m 1 --warps_n 8 --unroll 4 --repeat_m 8 --repeat_n 2\n";
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

        if (arg == "--m") { need_value("--m"); opts.m = std::stoll(argv[++i]); }
        else if (arg == "--n") { need_value("--n"); opts.n = std::stoll(argv[++i]); }
        else if (arg == "--k") { need_value("--k"); opts.k = std::stoll(argv[++i]); }
        else if (arg == "--warmup") { need_value("--warmup"); opts.warmup_iters = std::stoi(argv[++i]); }
        else if (arg == "--iters") { need_value("--iters"); opts.iters = std::stoi(argv[++i]); }
        else if (arg == "--warps_m") { need_value("--warps_m"); opts.block_warps_m = std::stoi(argv[++i]); }
        else if (arg == "--warps_n") { need_value("--warps_n"); opts.block_warps_n = std::stoi(argv[++i]); }
        else if (arg == "--unroll") { need_value("--unroll"); opts.unroll_k = std::stoi(argv[++i]); }
        else if (arg == "--repeat_m") { need_value("--repeat_m"); opts.repeat_m = std::stoi(argv[++i]); }
        else if (arg == "--repeat_n") { need_value("--repeat_n"); opts.repeat_n = std::stoi(argv[++i]); }
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

    if (opts.m <= 0 || opts.n <= 0 || opts.k <= 0 || opts.warmup_iters < 0 || opts.iters <= 0)
    {
        std::cerr << "All sizes must be positive, warmup must be non-negative, and iters must be positive."
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return opts;
}

int main(int argc, char** argv)
{
    const Options opts = parse_args(argc, argv);

    const int64_t chunk_k = 16 * opts.unroll_k;
    if (opts.k % chunk_k != 0) {
        std::cerr << "K must be divisible by 16 * unroll_k (" << chunk_k << ")" << std::endl;
        return 1;
    }

    const size_t a_elems = static_cast<size_t>(opts.m) * opts.k;
    const size_t b_elems = static_cast<size_t>(opts.k) * opts.n;
    const size_t c_elems = static_cast<size_t>(opts.m) * opts.n;

    // fp16: 2 bytes, fp8: 1 byte
    const double matrix_bytes_total = static_cast<double>(a_elems) * 2.0 +
                                      static_cast<double>(b_elems) * 1.0 +
                                      static_cast<double>(c_elems) * 2.0;
    const double gemm_flops = 2.0 * static_cast<double>(opts.m) * opts.n * opts.k;

    half* d_a = nullptr;
    uint8_t* d_b_prepacked = nullptr;
    half* d_c = nullptr;

    CHECK_HIP(hipMalloc(&d_a, a_elems * sizeof(half)));
    CHECK_HIP(hipMalloc(&d_b_prepacked, b_elems * sizeof(uint8_t)));
    CHECK_HIP(hipMalloc(&d_c, c_elems * sizeof(half)));

    CHECK_HIP(hipMemset(d_a, 0, a_elems * sizeof(half)));
    CHECK_HIP(hipMemset(d_b_prepacked, 0, b_elems * sizeof(uint8_t)));
    CHECK_HIP(hipMemset(d_c, 0, c_elems * sizeof(half)));

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    CHECK_HIP(hipStreamCreate(&stream));
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipStreamSynchronize(stream));

    for (int i = 0; i < opts.warmup_iters; ++i)
    {
        const bool launched = launch_scaled_mm(
            d_a, d_b_prepacked, nullptr, nullptr, d_c,
            opts.m, opts.n, opts.k,
            opts.k, opts.n,
            0, 0,
            opts.block_warps_m, opts.block_warps_n, opts.unroll_k, opts.repeat_m, opts.repeat_n,
            1, // b_dtype=1 (fp8e5m2)
            stream
        );
        if (!launched) {
            std::cerr << "Failed to launch kernel: unsupported config" << std::endl;
            return EXIT_FAILURE;
        }
    }
    CHECK_HIP(hipStreamSynchronize(stream));

    std::vector<float> timings_ms;
    timings_ms.reserve(static_cast<size_t>(opts.iters));
    for (int i = 0; i < opts.iters; ++i)
    {
        CHECK_HIP(hipEventRecord(start, stream));
        launch_scaled_mm(
            d_a, d_b_prepacked, nullptr, nullptr, d_c,
            opts.m, opts.n, opts.k,
            opts.k, opts.n,
            0, 0,
            opts.block_warps_m, opts.block_warps_n, opts.unroll_k, opts.repeat_m, opts.repeat_n,
            1, // b_dtype=1 (fp8e5m2)
            stream
        );
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
    const double avg_gbps = matrix_bytes_total / static_cast<double>(avg_ms) / 1.0e6;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "scaled_mm benchmark\n";
    std::cout << "  problem: m=" << opts.m << " n=" << opts.n << " k=" << opts.k << "\n";
    std::cout << "  config: warps_m=" << opts.block_warps_m << " warps_n=" << opts.block_warps_n
              << " unroll_k=" << opts.unroll_k << " repeat_m=" << opts.repeat_m << " repeat_n=" << opts.repeat_n << "\n";
    std::cout << "  memory footprint: " << (matrix_bytes_total / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "  avg ms: " << avg_ms << "\n";
    std::cout << "  min ms: " << min_ms << "\n";
    std::cout << "  max ms: " << max_ms << "\n";
    std::cout << "  avg TFLOP/s: " << avg_tflops << "\n";
    std::cout << "  avg GB/s: " << avg_gbps << std::endl;

    CHECK_HIP(hipEventDestroy(stop));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(d_c));
    CHECK_HIP(hipFree(d_b_prepacked));
    CHECK_HIP(hipFree(d_a));

    return EXIT_SUCCESS;
}
