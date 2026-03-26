#ifndef NO_PYTORCH
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#endif

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <optional>

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

template <int kBlockWarpsM,
          int kBlockWarpsN,
          int kUnrollK,
          int kRepeatM,
          int kRepeatN>
__global__ void mm_fp16_kernel(
    const half* __restrict__ const a,
    const half* __restrict__ const b_prepacked,
    const half* __restrict__ const bias,
    half* __restrict__ const c,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int has_bias)
{
    constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
    constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;

    // K0xMxK1 layout for A matrix (no extra LDS padding).
    // K1 = 8 for fp16: enables vec8 LDS reads (like CK)
    constexpr int kK1 = 8;
    // K0 = 16 / 8 = 2
    constexpr int kK0 = kWmmaK / kK1;
    constexpr int kAStrideK1 = kK1;
    constexpr int kShASize = kUnrollK * kK0 * kBlockM * kAStrideK1;

    // K0xNxK1 layout for B matrix
    constexpr int kBStrideK1 = kK1;
    constexpr int kShBSize = kUnrollK * kK0 * kBlockN * kBStrideK1;

    // C-shuffle epilogue reuses sh_a and sh_b memory. Each warp needs 16*16 halfs.
    constexpr int kCStride = kWmmaN;  // 16 elements per row

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

    const int tid = static_cast<int>(threadIdx.x);
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
    const auto sh_a_row_ptr = [&](const int stage, const int k0, const int m) -> half* {
        const int idx = ((stage * kK0 + k0) * kBlockM + m) * kAStrideK1;
        return &sh.ab.a[idx];
    };

    const auto load_a_lds_k0mk1 = [&](const int stage, const int64_t kk) -> void {
        // WSGR ownership: only A-owner waves issue A global->LDS stores.
        if constexpr (kUseWsgrAStoreOwnership) {
            if (wave_id >= kAOwnerWaves) return;
        }

        const int a_owner_tid = kUseWsgrAStoreOwnership ? (wave_id * kWaveSize + lane) : tid;

        // Physical LDS space is traversed directly.
        #pragma unroll
        for (int v = 0; v < kAVecsPerOwnerThread; ++v) {
            const int vec_idx = a_owner_tid + v * kAOwnerThreads;
            if (vec_idx >= kAVecs) continue;

            // Decode vec_idx to [k0][m_phys].
            const int k0 = vec_idx / kBlockM;
            const int m_phys = vec_idx % kBlockM;

            const int64_t a_row = block_m + m_phys;
            const int64_t a_k = kk + k0 * kK1; // Start K position for this K0 slice
            const half* __restrict__ const a_src = a + a_row * K + a_k;

            half* __restrict__ const sh_a_dst = sh_a_row_ptr(stage, k0, m_phys);
            *reinterpret_cast<uint4*>(sh_a_dst) = *reinterpret_cast<const uint4*>(a_src);
        }
    };

    // Loading B: K0xNxK1 layout
    constexpr int kBVecs = kK0 * kBlockN;
    constexpr int kBVecsPerThread = (kBVecs + kThreads - 1) / kThreads;
    const auto sh_b_row_ptr = [&](const int stage, const int k0, const int n) -> half* {
        const int idx = ((stage * kK0 + k0) * kBlockN + n) * kBStrideK1;
        return &sh.ab.b[idx];
    };

    const auto load_b_lds_k0nk1 = [&](const int stage, const int64_t kk) -> void {
        #pragma unroll
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec_idx = tid + v * kThreads;
            if (vec_idx >= kBVecs) continue;

            const int k0 = vec_idx / kBlockN;
            const int n_phys = vec_idx % kBlockN;

            const int64_t b_col = block_n + n_phys;
            const int64_t b_k = kk + k0 * kK1;

            // The python prepack stores B in (K/8, N, 8) with logical N order.
            // Adjacent b_row values still access adjacent 8-element (uint4) chunks.
            const half* __restrict__ const b_src = b_prepacked + b_k * N + b_col * kBStrideK1;

            half* __restrict__ const sh_b_dst = sh_b_row_ptr(stage, k0, n_phys);
            *reinterpret_cast<uint4*>(sh_b_dst) = *reinterpret_cast<const uint4*>(b_src);
        }
    };

    const auto load_b_global_vec = [&](const int vec_idx, const int64_t kk) -> uint4 {
        const int k0 = vec_idx / kBlockN;
        const int n_phys = vec_idx % kBlockN;

        const int64_t b_col = block_n + n_phys;
        const int64_t b_k = kk + k0 * kK1;
        const half* __restrict__ const b_src = b_prepacked + b_k * N + b_col * kBStrideK1;
        return *reinterpret_cast<const uint4*>(b_src);
    };

    const auto store_b_lds_vec = [&](const int stage, const int vec_idx, const uint4 value) -> void {
        const int k0 = vec_idx / kBlockN;
        const int n_phys = vec_idx % kBlockN;

        half* __restrict__ const sh_b_dst = sh_b_row_ptr(stage, k0, n_phys);
        *reinterpret_cast<uint4*>(sh_b_dst) = value;
    };

    // Pipeline setup
    constexpr int kChunkK = kWmmaK * kUnrollK;
    const int total_chunks = static_cast<int>(K / kChunkK);

    // WMMA compute lambda for one sub-iteration (one stage).
    // Register-tiling: load all B fragments once, then iterate rm with A loads.
    // Reduces LDS reads from (kRepeatM*kRepeatN)*(16+16) to kRepeatN*16 + kRepeatM*16.
    const auto wmma_compute_stage = [&](const int stage) -> void {
        if (wave_id >= kBlockWarpsM * kBlockWarpsN) return;

        using fp16x16_t = uint16_t __attribute__((ext_vector_type(16)));
        using fp32x8_t = float __attribute__((ext_vector_type(8)));

        const int lane_in_subgroup = lane % 16;

        // Pre-load all B fragments (one per rn tile)
        half all_reg_b[kRepeatN][16];
        #pragma unroll
        for (int rn = 0; rn < kRepeatN; ++rn) {
            const int tile_n = warp_n + rn * kBlockWarpsN;
            const int n_col = tile_n * kWmmaN + lane_in_subgroup;

            half reg_b[16];
            #pragma unroll
            for (int k0 = 0; k0 < kK0; ++k0) {
                const half* __restrict__ const sh_b_src = sh_b_row_ptr(stage, k0, n_col);
                #pragma unroll
                for (int k1 = 0; k1 < kK1; ++k1) {
                    reg_b[k0 * kK1 + k1] = sh_b_src[k1];
                }
            }
            *reinterpret_cast<fp16x16_t*>(&all_reg_b[rn]) = *reinterpret_cast<fp16x16_t*>(&reg_b);
        }

        // For each rm: load A once, compute all rn tiles with cached B
        #pragma unroll
        for (int rm = 0; rm < kRepeatM; ++rm) {
            const int tile_m = warp_m + rm * kBlockWarpsM;
            const int m_row = tile_m * kWmmaM + lane_in_subgroup;

            half reg_a[16];
            #pragma unroll
            for (int k0 = 0; k0 < kK0; ++k0) {
                const half* __restrict__ const sh_a_src = sh_a_row_ptr(stage, k0, m_row);
                #pragma unroll
                for (int k1 = 0; k1 < kK1; ++k1) {
                    reg_a[k0 * kK1 + k1] = sh_a_src[k1];
                }
            }

            const fp16x16_t a_frag = *reinterpret_cast<const fp16x16_t*>(reg_a);

            #pragma unroll
            for (int rn = 0; rn < kRepeatN; ++rn) {
                const int repeat_idx = rm * kRepeatN + rn;
                const fp16x16_t b_frag = *reinterpret_cast<const fp16x16_t*>(all_reg_b[rn]);
                fp32x8_t c_frag = *reinterpret_cast<fp32x8_t*>(&acc[repeat_idx][0]);

                c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);

                *reinterpret_cast<fp32x8_t*>(&acc[repeat_idx][0]) = c_frag;
            }
        }
    };

    // Prologue: load first chunk into LDS
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
        constexpr bool kBPrefetch =
            kBlockWarpsM == 1 &&
            kBlockWarpsN == 8 &&
            kUnrollK == 2 &&
            kRepeatM == 8 &&
            kRepeatN == 2;

        if constexpr (kBPrefetch) {
            static_assert(kBVecsPerThread == 2);
            static_assert(kBVecs == 2 * kThreads);

            uint4 b_stage0_prefetch0;
            uint4 b_stage0_prefetch1;
            uint4 b_stage1_prefetch0;
            uint4 b_stage1_prefetch1;
            const bool has_next = iter_idx + 1 < total_chunks;
            const int64_t k_next = static_cast<int64_t>(iter_idx + 1) * kChunkK;
            const int vec_idx0 = tid;
            const int vec_idx1 = tid + kThreads;

            if (has_next) {
                asm volatile("s_setprio 1" ::: "memory");
                b_stage0_prefetch0 = load_b_global_vec(vec_idx0, k_next);
                b_stage0_prefetch1 = load_b_global_vec(vec_idx1, k_next);
                b_stage1_prefetch0 = load_b_global_vec(vec_idx0, k_next + kWmmaK);
                b_stage1_prefetch1 = load_b_global_vec(vec_idx1, k_next + kWmmaK);
                asm volatile("s_setprio 0" ::: "memory");
            }

            wmma_compute_stage(0);
            wmma_compute_stage(1);
            __syncthreads();

            if (has_next) {
                store_b_lds_vec(0, vec_idx0, b_stage0_prefetch0);
                store_b_lds_vec(0, vec_idx1, b_stage0_prefetch1);
                store_b_lds_vec(1, vec_idx0, b_stage1_prefetch0);
                store_b_lds_vec(1, vec_idx1, b_stage1_prefetch1);

                asm volatile("s_setprio 1" ::: "memory");
                load_a_lds_k0mk1(0, k_next);
                load_a_lds_k0mk1(1, k_next + kWmmaK);
                asm volatile("s_setprio 0" ::: "memory");
                __syncthreads();
            }
        } else {
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
    }

    // Epilogue: C-Shuffle - write output with coalesced vec8 stores
    // Use LDS to transpose from column-major (WMMA layout) to row-major (coalesced)
    if (wave_id < kBlockWarpsM * kBlockWarpsN) {
        half* __restrict__ const sh_c = sh.c[wave_id][0];
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
                    sh_c[row_phys * kCStride + col] = __float2half_rn(acc[repeat_idx][acc_idx]);
                }

                // Wave executes in lockstep (SIMT), so all writes complete before reads
                // No explicit barrier needed within a wave

                // Step 2: Read from LDS in row-major order for coalesced global write
                // 32 threads -> 16 rows, 2 threads per row, each handles 8 columns
                const int read_row = lane / 2;
                const int read_row_phys = c_row_logi_to_phys_16(read_row);
                const int col_half = lane % 2;  // 0 = cols 0-7, 1 = cols 8-15
                const int read_col_base = col_half * 8;

                const int64_t out_row = tile_m_base + read_row;
                const int64_t out_col = tile_n_base + read_col_base;

                half* __restrict__ const out_ptr = c + out_row * N + out_col;
                half* __restrict__ const h = sh_c + read_row_phys * kCStride + read_col_base;

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

template <int M, int N, int U, int RM, int RN>
struct ConfigTag {
    static constexpr int kBlockWarpsM = M;
    static constexpr int kBlockWarpsN = N;
    static constexpr int kUnrollK = U;
    static constexpr int kRepeatM = RM;
    static constexpr int kRepeatN = RN;
};

extern "C" bool launch_mm_fp16(
    const half* const a,
    const half* const b_prepacked,
    const half* const bias,
    half* const c,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int has_bias,
    const int block_warps_m,
    const int block_warps_n,
    const int unroll_k,
    const int repeat_m,
    const int repeat_n,
    hipStream_t stream)
{
    const auto launch = [&](const auto tag) -> void {
        constexpr int kBlockWarpsM = decltype(tag)::kBlockWarpsM;
        constexpr int kBlockWarpsN = decltype(tag)::kBlockWarpsN;
        constexpr int kUnrollK = decltype(tag)::kUnrollK;
        constexpr int kRepeatM = decltype(tag)::kRepeatM;
        constexpr int kRepeatN = decltype(tag)::kRepeatN;
        constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
        constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;

        constexpr int kThreadsPerBlock = kWaveSize * kBlockWarpsM * kBlockWarpsN;
        const dim3 block(kThreadsPerBlock, 1, 1);
        const dim3 grid(static_cast<uint32_t>(N / kBlockN), static_cast<uint32_t>(M / kBlockM), 1);

        hipLaunchKernelGGL(
            (mm_fp16_kernel<kBlockWarpsM, kBlockWarpsN, kUnrollK, kRepeatM, kRepeatN>),
            grid, block, 0, stream,
            a, b_prepacked, bias, c,
            M, N, K,
            has_bias);
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

    // Autotune configs
    // Format: (warps_m, warps_n, unroll_k, repeat_m, repeat_n)
    return
        try_launch(ConfigTag<1, 1, 2, 1, 2>{}) ||
        try_launch(ConfigTag<1, 1, 4, 1, 2>{}) ||
        try_launch(ConfigTag<1, 2, 2, 1, 2>{}) ||
        try_launch(ConfigTag<1, 2, 4, 1, 2>{}) ||
        try_launch(ConfigTag<1, 4, 2, 1, 2>{}) ||
        try_launch(ConfigTag<1, 4, 4, 1, 2>{}) ||
        try_launch(ConfigTag<1, 8, 2, 1, 2>{}) ||
        try_launch(ConfigTag<1, 8, 4, 1, 2>{}) ||
        try_launch(ConfigTag<1, 1, 2, 2, 2>{}) ||
        try_launch(ConfigTag<1, 1, 4, 2, 2>{}) ||
        try_launch(ConfigTag<1, 1, 2, 4, 4>{}) ||
        try_launch(ConfigTag<1, 1, 4, 4, 4>{}) ||
        try_launch(ConfigTag<1, 2, 2, 2, 2>{}) ||
        try_launch(ConfigTag<1, 2, 4, 2, 2>{}) ||
        try_launch(ConfigTag<2, 1, 2, 2, 2>{}) ||
        try_launch(ConfigTag<2, 1, 4, 2, 2>{}) ||
        try_launch(ConfigTag<1, 4, 2, 4, 2>{}) ||
        try_launch(ConfigTag<1, 4, 4, 4, 2>{}) ||
        try_launch(ConfigTag<1, 8, 2, 8, 2>{}) ||
        try_launch(ConfigTag<1, 8, 4, 8, 2>{}) ||
        try_launch(ConfigTag<2, 2, 2, 4, 4>{}) ||
        try_launch(ConfigTag<2, 2, 4, 4, 4>{}) ||
        try_launch(ConfigTag<2, 4, 2, 4, 2>{}) ||
        try_launch(ConfigTag<2, 4, 4, 4, 2>{}) ||
        try_launch(ConfigTag<2, 4, 2, 4, 4>{}) ||
        try_launch(ConfigTag<2, 4, 4, 4, 4>{}) ||
        try_launch(ConfigTag<4, 2, 2, 2, 4>{}) ||
        try_launch(ConfigTag<4, 2, 4, 2, 4>{}) ||
        false;
}

#ifndef NO_PYTORCH
void mm_fp16(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b_prepacked,
    const std::optional<torch::stable::Tensor>& bias,
    torch::stable::Tensor& c,
    const int64_t block_warps_m,
    const int64_t block_warps_n,
    const int64_t unroll_k,
    const int64_t repeat_m,
    const int64_t repeat_n)
{
    STD_TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    STD_TORCH_CHECK(b_prepacked.is_cuda(), "b_prepacked must be a CUDA tensor");
    STD_TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
    const auto device_index = a.get_device_index();
    STD_TORCH_CHECK(b_prepacked.get_device_index() == device_index, "b_prepacked must be on the same device as a");
    STD_TORCH_CHECK(c.get_device_index() == device_index, "c must be on the same device as a");

    STD_TORCH_CHECK(a.scalar_type() == torch::stable::ScalarType::Half, "a must be float16");
    STD_TORCH_CHECK(b_prepacked.scalar_type() == torch::stable::ScalarType::Half, "b_prepacked must be float16");
    STD_TORCH_CHECK(c.scalar_type() == torch::stable::ScalarType::Half, "c must be float16");

    STD_TORCH_CHECK(a.dim() == 2, "a must be 2D");
    STD_TORCH_CHECK(b_prepacked.dim() == 3, "b_prepacked must be 3D (K/8, N, 8)");
    STD_TORCH_CHECK(c.dim() == 2, "c must be 2D");

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = b_prepacked.size(1);
    STD_TORCH_CHECK(K % kWmmaK == 0, "K must be divisible by 16");
    STD_TORCH_CHECK(b_prepacked.size(0) == K / 8, "b_prepacked.shape[0] must equal K/8 (", K / 8, ")");
    STD_TORCH_CHECK(b_prepacked.size(2) == 8, "b_prepacked.shape[2] must be 8");
    STD_TORCH_CHECK(c.size(0) == M, "c.shape[0] must equal M");
    STD_TORCH_CHECK(c.size(1) == N, "c.shape[1] must equal N");

    // Contiguous fast path requirements
    STD_TORCH_CHECK(a.stride(0) == K, "a.stride(0) must equal K (", K, ")");
    STD_TORCH_CHECK(a.stride(1) == 1, "a.stride(1) must be 1");
    STD_TORCH_CHECK(b_prepacked.stride(0) == N * 8, "b_prepacked.stride(0) must equal N*8 (", N * 8, ")");
    STD_TORCH_CHECK(b_prepacked.stride(1) == 8, "b_prepacked.stride(1) must be 8");
    STD_TORCH_CHECK(b_prepacked.stride(2) == 1, "b_prepacked.stride(2) must be 1");
    STD_TORCH_CHECK(c.stride(0) == N, "c.stride(0) must equal N (", N, ")");
    STD_TORCH_CHECK(c.stride(1) == 1, "c.stride(1) must be 1");

    if (bias.has_value()) {
        STD_TORCH_CHECK(bias.has_value(), "bias must be provided when has_bias=True");
        const auto& bias_t = *bias;
        STD_TORCH_CHECK(bias_t.is_cuda(), "bias must be a CUDA tensor");
        STD_TORCH_CHECK(bias_t.get_device_index() == device_index, "bias must be on the same device as a");
        STD_TORCH_CHECK(bias_t.scalar_type() == torch::stable::ScalarType::Half, "bias must be float16");
        STD_TORCH_CHECK(bias_t.numel() == N, "bias must have N elements");
    }

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    const half* const a_ptr = reinterpret_cast<const half*>(a.const_data_ptr());
    const half* const b_ptr = reinterpret_cast<const half*>(b_prepacked.const_data_ptr());
    const half* const bias_ptr = bias.has_value() ? reinterpret_cast<const half*>(bias->const_data_ptr()) : nullptr;
    half* const c_ptr = reinterpret_cast<half*>(c.mutable_data_ptr());

    const auto is_aligned_16 = [](const void* const p) {
        return (reinterpret_cast<uintptr_t>(p) & 0xFu) == 0u;
    };
    STD_TORCH_CHECK(is_aligned_16(a_ptr), "a data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(b_ptr), "b_prepacked data pointer must be 16-byte aligned");
    STD_TORCH_CHECK(is_aligned_16(c_ptr), "c data pointer must be 16-byte aligned");

    const int64_t block_m = kWmmaM * block_warps_m * repeat_m;
    const int64_t block_n = kWmmaN * block_warps_n * repeat_n;
    const int64_t chunk_k = kWmmaK * unroll_k;
    STD_TORCH_CHECK(M % block_m == 0, "M (", M, ") must be divisible by kBlockM (", block_m, ")");
    STD_TORCH_CHECK(N % block_n == 0, "N (", N, ") must be divisible by kBlockN (", block_n, ")");
    STD_TORCH_CHECK(K % chunk_k == 0, "K (", K, ") must be divisible by kChunkK (", chunk_k, ")");

    const int64_t threads_per_block = kWaveSize * block_warps_m * block_warps_n;
    STD_TORCH_CHECK(threads_per_block <= 1024, "Block size exceeds HIP thread-per-block limit");

    const bool launched = launch_mm_fp16(
        a_ptr, b_ptr, bias_ptr, c_ptr,
        M, N, K,
        bias.has_value() ? 1 : 0,
        block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n,
        stream);
    STD_TORCH_CHECK(launched, "Unsupported config");

    const hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "mm_fp16 kernel launch failed: ", hipGetErrorString(launch_err));
}

STABLE_TORCH_LIBRARY(feather_ops, m)
{
    m.def(
        "mm_fp16("
        "Tensor a, "
        "Tensor b_prepacked, "
        "Tensor? bias, "
        "Tensor(a!) c, "
        "int block_warps_m, "
        "int block_warps_n, "
        "int unroll_k, "
        "int repeat_m, "
        "int repeat_n"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_ops, CUDA, m)
{
    m.impl("mm_fp16", TORCH_BOX(&mm_fp16));
}
#endif
