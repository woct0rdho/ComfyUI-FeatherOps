#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
// gfx11 uses wave32 - hardcode for consistent host/device behavior
// rocwmma::Constants::AMDGCN_WAVE_SIZE returns 64 during host compilation
constexpr int kWaveSize = 32;

// 16-row swizzle used by LDS physical mapping.
__device__ __forceinline__ constexpr int c_row_logi_to_phys_16(const int x)
{
    return ((x & 7) << 1) | ((x >> 3) & 1);
}

__device__ __forceinline__ constexpr int a_row_phys_to_logi_16(const int x)
{
    return ((x & 1) << 3) | ((x >> 1) & 7);
}

template <int kBlockWarpsM,
          int kBlockWarpsN,
          int kUnrollK,
          int kRepeatM,
          int kRepeatN>
__global__ void mm_kernel_fp16_prepacked_b(
    const half* const a,
    const half* const b_prepacked,
    const half* const bias,
    half* const c,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t stride_am,
    const int64_t stride_cm,
    const int has_bias)
{
    constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
    constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;
    static_assert(kBlockM % 16 == 0, "kBlockM must be a multiple of 16 (required by row swizzle)");
    static_assert(kBlockN % 16 == 0, "kBlockN must be a multiple of 16");

    // K0xMxK1 layout for A matrix (no extra LDS padding).
    // Apply row permutation on A store to improve LDS local-read banking while
    // keeping compact LDS footprint and 128-bit accesses.
    // K1 = 8 for fp16: enables vec8 LDS reads (like CK)
    constexpr int kK1 = 8;
    // K0 = kWmmaK / K1 = 16 / 8 = 2
    constexpr int kK0 = kWmmaK / kK1;
    constexpr int kAStrideK1 = kK1;
    constexpr int kShASize = kUnrollK * kK0 * kBlockM * kAStrideK1;

    // K0xNxK1 layout for B matrix
    constexpr int kBStrideK1 = kK1;
    constexpr int kShBSize = kUnrollK * kK0 * kBlockN * kBStrideK1;

    // C-shuffle epilogue reuses sh_a and sh_b memory. Each warp needs 16*16 halfs.
    constexpr int kCStride = kWmmaN;  // 16 halfs per row

    union SharedStorage {
        struct {
            half a[kShASize];
            half b[kShBSize];
        } ab;
        half c[kBlockWarpsM * kBlockWarpsN][kWmmaM][kCStride];
    };
    __shared__ __align__(16) SharedStorage sh;

    const int block_m = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_n = static_cast<int>(blockIdx.x) * kBlockN;

    const int tid = static_cast<int>(threadIdx.x) + static_cast<int>(threadIdx.y) * static_cast<int>(blockDim.x);
    constexpr int kThreads = kWaveSize * kBlockWarpsM * kBlockWarpsN;

    // Flattened wave mapping with 1D thread blocks.
    const int wave_id = tid / kWaveSize;
    const int warp_m = wave_id % kBlockWarpsM;
    const int warp_n = wave_id / kBlockWarpsM;
    const int lane = tid % kWaveSize;

    // Accumulator registers: 8 floats per WMMA tile in wave32 mode
    constexpr int kRepeatTiles = kRepeatM * kRepeatN;
    float acc[kRepeatTiles][8];
    #pragma unroll
    for (int r = 0; r < kRepeatTiles; ++r) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[r][i] = 0.0f;
        }
    }

    // Loading A: K0xMxK1 layout with physical/inverse row mapping and
    // wave-separated global-read ownership.
    constexpr int kAVecs = kK0 * kBlockM;
    // Use wave-separated ownership only when per-thread vec count stays small (<=4).
    // For large kBlockM (e.g. kRepeatM=8, kBlockM=256) WSGR causes too many VGPRs
    // for the A prefetch buffer, hurting occupancy.
    constexpr bool kUseWsgrAStoreOwnership =
        (kAVecs / (kBlockWarpsM * kWaveSize)) <= 4;
    constexpr int kAOwnerWaves =
        kUseWsgrAStoreOwnership ? kBlockWarpsM : (kBlockWarpsM * kBlockWarpsN);
    constexpr int kAOwnerThreads = kAOwnerWaves * kWaveSize;
    constexpr int kAVecsPerOwnerThread = (kAVecs + kAOwnerThreads - 1) / kAOwnerThreads;
    const auto a_row_phys_to_logi = [&](const int row_phys) -> int {
        return row_phys; // Currently we do not need A swizzle
    };
    const auto sh_a_row_ptr = [&](const int stage, const int k0, const int m) -> half* {
        const int idx = (((stage * kK0 + k0) * kBlockM + m) * kAStrideK1);
        return &sh.ab.a[idx];
    };

    const auto load_a_lds_k0mk1 = [&](const int stage, const int64_t kk) -> void {
        // WSGR ownership: only A-owner waves issue A global->LDS stores.
        if constexpr (kUseWsgrAStoreOwnership) {
            if (wave_id >= kAOwnerWaves) return;
        }

        const int a_owner_tid = kUseWsgrAStoreOwnership ? (wave_id * kWaveSize + lane) : tid;

        // Physical LDS space is traversed directly; global logical row is obtained by inverse map.
        #pragma unroll
        for (int v = 0; v < kAVecsPerOwnerThread; ++v) {
            const int vec_idx = a_owner_tid + v * kAOwnerThreads;
            if (vec_idx >= kAVecs) continue;

            // Decode vec_idx to [k0][m_phys].
            const int k0 = vec_idx / kBlockM;
            const int m_phys = vec_idx % kBlockM;
            const int m_logi = a_row_phys_to_logi(m_phys);

            const int64_t a_row = block_m + m_logi;
            const int64_t a_k = kk + k0 * kK1; // Start K position for this K0 slice
            half* const sh_a_dst = sh_a_row_ptr(stage, k0, m_phys);

            const half* const a_ptr = a + a_row * stride_am + a_k;
            *reinterpret_cast<uint4*>(sh_a_dst) = *reinterpret_cast<const uint4*>(a_ptr);
        }
    };

    // Loading B: K0xNxK1 layout
    constexpr int kBVecs = kK0 * kBlockN;
    constexpr int kBVecsPerThread = (kBVecs + kThreads - 1) / kThreads;
    const auto sh_b_row_ptr = [&](const int stage, const int k0, const int n) -> half* {
        const int idx = (((stage * kK0 + k0) * kBlockN + n) * kBStrideK1);
        return &sh.ab.b[idx];
    };

    const auto load_b_lds_k0nk1 = [&](const int stage, const int64_t kk) -> void {
        #pragma unroll
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec_idx = tid + v * kThreads;
            if (vec_idx >= kBVecs) continue;

            const int k0 = vec_idx / kBlockN;
            const int n_phys = vec_idx % kBlockN;

            const int64_t ktile = kk / kWmmaK;
            const int64_t n_col = block_n + n_phys;

            half* const sh_b_dst = sh_b_row_ptr(stage, k0, n_phys);

            // The python prepack stores B in [K/16, 2, N, 8] with logical N order.
            // Adjacent n_col values still access adjacent 8-element (uint4) chunks.
            const int64_t gidx = ktile * (2 * N * 8) + k0 * (N * 8) + n_col * 8;
            const half* const b_src = b_prepacked + gidx;

            *reinterpret_cast<uint4*>(sh_b_dst) = *reinterpret_cast<const uint4*>(b_src);
        }
    };

    // Pipeline setup
    constexpr int kChunkK = kWmmaK * kUnrollK;
    const int total_chunks = static_cast<int>(K / kChunkK);

    // WMMA compute lambda for one sub-iteration (one stage).
    // Register-tiling: load all B fragments once, then iterate rm with A loads.
    // Reduces LDS reads from (kRepeatM*kRepeatN)*(16+16) to kRepeatN*16 + kRepeatM*16.
    const auto wmma_compute_stage = [&](const int stage) -> void {
        if (wave_id >= kBlockWarpsM * kBlockWarpsN) return;

        using fp16x16_t = _Float16 __attribute__((ext_vector_type(16)));
        using float8_t = float __attribute__((ext_vector_type(8)));

        const int lane_in_subgroup = lane % 16;

        // Pre-load all B fragments (one per rn tile)
        _Float16 all_reg_b[kRepeatN][16];
        #pragma unroll
        for (int rn = 0; rn < kRepeatN; ++rn) {
            const int tile_n = warp_n + rn * kBlockWarpsN;
            const int n_logi = tile_n * kWmmaN + lane_in_subgroup;

            const int n_phys = n_logi;

            _Float16 reg_b_fp16[16];
            #pragma unroll
            for (int k0 = 0; k0 < kK0; ++k0) {
                const half* const sh_b_src = sh_b_row_ptr(stage, k0, n_phys);
                #pragma unroll
                for (int k1 = 0; k1 < kK1; ++k1) {
                    reg_b_fp16[k0 * kK1 + k1] = static_cast<_Float16>(sh_b_src[k1]);
                }
            }
            *reinterpret_cast<fp16x16_t*>(&all_reg_b[rn]) = *reinterpret_cast<fp16x16_t*>(&reg_b_fp16);
        }

        // For each rm: load A once, compute all rn tiles with cached B
        #pragma unroll
        for (int rm = 0; rm < kRepeatM; ++rm) {
            const int tile_m = warp_m + rm * kBlockWarpsM;
            const int m_row = tile_m * kWmmaM + lane_in_subgroup;

            _Float16 reg_a_fp16[16];
            #pragma unroll
            for (int k0 = 0; k0 < kK0; ++k0) {
                const half* const sh_a_src = sh_a_row_ptr(stage, k0, m_row);
                #pragma unroll
                for (int k1 = 0; k1 < kK1; ++k1) {
                    reg_a_fp16[k0 * kK1 + k1] = static_cast<_Float16>(sh_a_src[k1]);
                }
            }

            const fp16x16_t a_frag = *reinterpret_cast<const fp16x16_t*>(reg_a_fp16);

            #pragma unroll
            for (int rn = 0; rn < kRepeatN; ++rn) {
                const int repeat_idx = rm * kRepeatN + rn;
                const fp16x16_t b_frag = *reinterpret_cast<const fp16x16_t*>(all_reg_b[rn]);
                float8_t c_frag = *reinterpret_cast<float8_t*>(&acc[repeat_idx][0]);

                c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);

                *reinterpret_cast<float8_t*>(&acc[repeat_idx][0]) = c_frag;
            }
        }
    };

    // Prologue: load first chunk into LDS using monolithic loads
    #pragma unroll
    for (int u = 0; u < kUnrollK; ++u) {
        const int64_t k = static_cast<int64_t>(u) * kWmmaK;
        load_a_lds_k0mk1(u, k);
    }
    #pragma unroll
    for (int u = 0; u < kUnrollK; ++u) {
        const int64_t k = static_cast<int64_t>(u) * kWmmaK;
        load_b_lds_k0nk1(u, k);
    }
    __syncthreads();

    // Main loop
    for (int iter_idx = 0; iter_idx < total_chunks; ++iter_idx) {
        #pragma unroll
        for (int u = 0; u < kUnrollK; ++u) {
            wmma_compute_stage(u);
        }
        __syncthreads();

        if (iter_idx + 1 < total_chunks) {
            const int64_t k_next = static_cast<int64_t>(iter_idx + 1) * kChunkK;
            asm volatile("s_setprio 1" ::: "memory");
            #pragma unroll
            for (int u = 0; u < kUnrollK; ++u) {
                const int64_t k = k_next + static_cast<int64_t>(u) * kWmmaK;
                load_a_lds_k0mk1(u, k);
            }
            #pragma unroll
            for (int u = 0; u < kUnrollK; ++u) {
                const int64_t k = k_next + static_cast<int64_t>(u) * kWmmaK;
                load_b_lds_k0nk1(u, k);
            }
            asm volatile("s_setprio 0" ::: "memory");
            __syncthreads();
        }
    }

    // Epilogue: C-Shuffle - write output with coalesced vec8 stores
    // Use LDS to transpose from column-major (WMMA layout) to row-major (coalesced)
    if (wave_id < kBlockWarpsM * kBlockWarpsN) {
        half* const sh_c = sh.c[wave_id][0];
        const int subgroup = lane / 16;
        const int lane_in_subgroup = lane % 16;

        #pragma unroll
        for (int rm = 0; rm < kRepeatM; ++rm) {
            #pragma unroll
            for (int rn = 0; rn < kRepeatN; ++rn) {
                const int repeat_idx = rm * kRepeatN + rn;
                const int tile_m = warp_m + rm * kBlockWarpsM;
                const int tile_n = warp_n + rn * kBlockWarpsN;
                const int64_t tile_m_base = block_m + tile_m * kWmmaM;
                const int64_t tile_n_base = block_n + tile_n * kWmmaN;

                // Step 1: Write acc to LDS in column-major order (WMMA layout)
                // Each thread writes 8 values to one column
                const int col = lane_in_subgroup;
                #pragma unroll
                for (int acc_idx = 0; acc_idx < 8; ++acc_idx) {
                    // v_wmma_f32_16x16x16_f16_w32 layout:
                    // subgroup 0 (lanes 0-15) holds even rows: 0, 2, 4, 6, 8, 10, 12, 14
                    // subgroup 1 (lanes 16-31) holds odd rows: 1, 3, 5, 7, 9, 11, 13, 15
                    const int row_logi = acc_idx * 2 + subgroup;
                    const int row_phys = c_row_logi_to_phys_16(row_logi);
                    half val = __float2half_rn(acc[repeat_idx][acc_idx]);
                    sh_c[row_phys * kCStride + col] = val;
                }

                // Wave executes in lockstep (SIMT), so all writes complete before reads
                // No explicit barrier needed within a wave

                // Step 2: Read from LDS in row-major order for coalesced global write
                // 32 threads -> 16 rows, 2 threads per row, each handles 8 columns
                const int read_row = lane / 2;
                const int read_row_phys = c_row_logi_to_phys_16(read_row);
                const int col_half = lane % 2;
                const int read_col_base = col_half * 8;

                const int64_t out_row = tile_m_base + read_row;
                const int64_t out_col = tile_n_base + read_col_base;

                half* const out_ptr = c + out_row * stride_cm + out_col;
                half* const h = sh_c + read_row_phys * kCStride + read_col_base;

                if (has_bias) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        h[i] = __hadd(h[i], bias[out_col + i]);
                    }
                }

                *reinterpret_cast<uint4*>(out_ptr) = *reinterpret_cast<uint4*>(h);
            }
        }
    }
}

} // namespace

// Config tag for kernel (no vec_a/vec_b params - always uses vec8 A, vec8 B)
template <int M, int N, int U, int RM, int RN>
struct ConfigTag {
    static constexpr int kBlockWarpsM = M;
    static constexpr int kBlockWarpsN = N;
    static constexpr int kUnrollK = U;
    static constexpr int kRepeatM = RM;
    static constexpr int kRepeatN = RN;
};

torch::Tensor mm_fp16(
    const torch::Tensor& a,
    const torch::Tensor& b_prepacked,
    const torch::Tensor& bias,
    const bool has_bias,
    const int64_t block_warps_m,
    const int64_t block_warps_n,
    const int64_t unroll_k,
    const int64_t repeat_m,
    const int64_t repeat_n)
{
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b_prepacked.is_cuda(), "b_prepacked must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be float16");
    TORCH_CHECK(b_prepacked.scalar_type() == at::kHalf, "b_prepacked must be float16");
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b_prepacked.dim() == 4, "b_prepacked must be 4D [K/16, 2, N, 8]");

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = b_prepacked.size(2);
    TORCH_CHECK(K % kWmmaK == 0, "K must be divisible by 16");
    TORCH_CHECK(b_prepacked.size(0) == K / kWmmaK,
        "b_prepacked.shape[0] must equal K/16 (", K / kWmmaK, ")");
    TORCH_CHECK(b_prepacked.size(1) == 2, "b_prepacked.shape[1] must be 2");
    TORCH_CHECK(b_prepacked.size(3) == 8, "b_prepacked.shape[3] must be 8");

    // Contiguous fast path requirements
    TORCH_CHECK(a.stride(1) == 1, "a must be row-contiguous (stride(1) == 1)");
    TORCH_CHECK(b_prepacked.stride(3) == 1, "b_prepacked last dim must be contiguous");

    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.numel() == N, "bias must have N elements");
        TORCH_CHECK(bias.scalar_type() == at::kHalf, "bias must be float16");
    }

    auto c = torch::empty({M, N}, a.options().dtype(at::kHalf));

    const half* const a_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* const b_ptr = reinterpret_cast<const half*>(b_prepacked.data_ptr<at::Half>());
    auto stream = at::cuda::getCurrentCUDAStream();
    const half* const bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;
    half* const c_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const auto is_aligned_16 = [](const void* const p) {
        return (reinterpret_cast<uintptr_t>(p) & 0xFu) == 0u;
    };
    TORCH_CHECK(is_aligned_16(a_ptr), "a data pointer must be 16-byte aligned");
    TORCH_CHECK(is_aligned_16(b_ptr), "b_prepacked data pointer must be 16-byte aligned");
    TORCH_CHECK(is_aligned_16(c_ptr), "c data pointer must be 16-byte aligned");

    const auto launch = [&](const auto tag) -> void {
        constexpr int kBlockWarpsM = decltype(tag)::kBlockWarpsM;
        constexpr int kBlockWarpsN = decltype(tag)::kBlockWarpsN;
        constexpr int kUnrollK = decltype(tag)::kUnrollK;
        constexpr int kRepeatM = decltype(tag)::kRepeatM;
        constexpr int kRepeatN = decltype(tag)::kRepeatN;
        constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
        constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;

        TORCH_CHECK(M % kBlockM == 0,
            "M (", M, ") must be divisible by kBlockM (", kBlockM, ")");
        TORCH_CHECK(N % kBlockN == 0,
            "N (", N, ") must be divisible by kBlockN (", kBlockN, ")");
        TORCH_CHECK(K % (kWmmaK * kUnrollK) == 0,
            "K (", K, ") must be divisible by kChunkK (", kWmmaK * kUnrollK, ")");

        constexpr int kThreadsPerBlock = kWaveSize * kBlockWarpsM * kBlockWarpsN;
        static_assert(kThreadsPerBlock <= 1024, "Block size exceeds HIP thread-per-block limit");
        const dim3 block(kThreadsPerBlock, 1, 1);
        const dim3 grid(
            static_cast<uint32_t>(N) / kBlockN,
            static_cast<uint32_t>(M) / kBlockM);

        hipLaunchKernelGGL(
            (mm_kernel_fp16_prepacked_b<kBlockWarpsM, kBlockWarpsN, kUnrollK, kRepeatM, kRepeatN>),
            grid, block, 0, stream.stream(),
            a_ptr, b_ptr, bias_ptr, c_ptr,
            M, N, K,
            a.stride(0), c.stride(0),
            has_bias ? 1 : 0);
    };

    const auto try_launch = [&](const auto tag) -> bool {
        if (block_warps_m == decltype(tag)::kBlockWarpsM &&
            block_warps_n == decltype(tag)::kBlockWarpsN &&
            unroll_k == decltype(tag)::kUnrollK &&
            repeat_m == decltype(tag)::kRepeatM &&
            repeat_n == decltype(tag)::kRepeatN) {
            launch(tag);
            return true;
        }
        return false;
    };

    // Autotune candidate configs
    // Format: (warps_m, warps_n, unroll_k, repeat_m, repeat_n)
    const bool launched =
        try_launch(ConfigTag<1, 1, 2, 2, 2>{}) ||
        try_launch(ConfigTag<1, 1, 2, 4, 4>{}) ||
        try_launch(ConfigTag<1, 1, 4, 2, 2>{}) ||
        try_launch(ConfigTag<1, 1, 4, 4, 4>{}) ||
        try_launch(ConfigTag<1, 1, 8, 2, 2>{}) ||
        try_launch(ConfigTag<1, 1, 8, 4, 4>{}) ||
        try_launch(ConfigTag<1, 2, 2, 2, 2>{}) ||
        try_launch(ConfigTag<1, 2, 4, 2, 2>{}) ||
        try_launch(ConfigTag<1, 2, 8, 2, 2>{}) ||
        try_launch(ConfigTag<1, 4, 2, 4, 2>{}) ||
        try_launch(ConfigTag<1, 4, 4, 4, 2>{}) ||
        try_launch(ConfigTag<1, 4, 8, 4, 2>{}) ||
        try_launch(ConfigTag<1, 8, 2, 8, 2>{}) ||
        try_launch(ConfigTag<1, 8, 4, 8, 2>{}) ||
        try_launch(ConfigTag<2, 1, 2, 2, 2>{}) ||
        try_launch(ConfigTag<2, 1, 4, 2, 2>{}) ||
        try_launch(ConfigTag<2, 1, 8, 2, 2>{}) ||
        try_launch(ConfigTag<2, 2, 2, 4, 4>{}) ||
        try_launch(ConfigTag<2, 2, 4, 4, 4>{}) ||
        try_launch(ConfigTag<2, 2, 8, 4, 4>{}) ||
        try_launch(ConfigTag<2, 4, 2, 4, 2>{}) ||
        try_launch(ConfigTag<2, 4, 2, 4, 4>{}) ||
        try_launch(ConfigTag<2, 4, 4, 4, 2>{}) ||
        try_launch(ConfigTag<2, 4, 4, 4, 4>{}) ||
        try_launch(ConfigTag<2, 4, 8, 4, 2>{}) ||
        try_launch(ConfigTag<4, 2, 2, 2, 4>{}) ||
        try_launch(ConfigTag<4, 2, 4, 2, 4>{}) ||
        try_launch(ConfigTag<4, 2, 8, 2, 4>{}) ||
        false;

    TORCH_CHECK(launched, "Unsupported config");
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    namespace py = pybind11;
    m.def(
        "mm_fp16",
        &mm_fp16,
        py::arg("a"),
        py::arg("b_prepacked"),
        py::arg("bias"),
        py::arg("has_bias"),
        py::arg("block_warps_m"),
        py::arg("block_warps_n"),
        py::arg("unroll_k"),
        py::arg("repeat_m"),
        py::arg("repeat_n"),
        "FP16 matmul (prepacked B [K/16,N,16]).");
}
