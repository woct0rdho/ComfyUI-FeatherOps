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

struct bfloat16_t {
    uint16_t data;

    __device__ __forceinline__ bfloat16_t() = default;

    __device__ __forceinline__ bfloat16_t(const float f) {
        const uint32_t u = *reinterpret_cast<const uint32_t*>(&f);
        data = static_cast<uint16_t>(u >> 16);
    }

    __device__ __forceinline__ operator float() const {
        const uint32_t u = static_cast<uint32_t>(data) << 16;
        return *reinterpret_cast<const float*>(&u);
    }
};

__device__ __forceinline__ bfloat16_t operator+(const bfloat16_t a, const bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) + static_cast<float>(b));
}

__device__ __forceinline__ bfloat16_t operator*(const bfloat16_t a, const bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) * static_cast<float>(b));
}

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
// gfx11 uses wave32 - hardcode for consistent host/device behavior
// rocwmma::Constants::AMDGCN_WAVE_SIZE returns 64 during host compilation
constexpr int kWaveSize = 32;

// Packed fp8e4m3fn -> fp16 conversion: converts 4 fp8 bytes in a uint32 to 4 fp16 values.
// Produces two uint32s in sequential half2 order: out_lo=[h1:h0], out_hi=[h3:h2].
// Ignores denormals (values with zero exponent map to small fp16 instead of zero).
__device__ __forceinline__ void fp8e4m3x4_to_half2x2(
    const uint32_t p, uint32_t& out_lo, uint32_t& out_hi)
{
    // p = [b3:b2:b1:b0], each byte is fp8e4m3fn.
    // Build byte pairs as [0:b1:0:b0] and [0:b3:0:b2] using V_PERM_B32.
    const uint32_t lo_pair = __builtin_amdgcn_perm(0u, p, 0x0c010c00u);
    const uint32_t hi_pair = __builtin_amdgcn_perm(0u, p, 0x0c030c02u);

    // For each 16-bit lane with byte x in [7:0]:
    // fp16 = (x << 7) + ((x & 0x80) << 7) + 0x2000
    // This form avoids explicit and_or masking and lowers to shift-add patterns.
    out_lo = (lo_pair << 7) + ((lo_pair & 0x00800080u) << 7) + 0x20002000u;
    out_hi = (hi_pair << 7) + ((hi_pair & 0x00800080u) << 7) + 0x20002000u;
}

// Packed fp8e5m2 -> fp16 conversion: converts 4 fp8 bytes in a uint32 to 4 fp16 values.
// Produces two uint32s in sequential half2 order: out_lo=[h1:h0], out_hi=[h3:h2].
// fp8e5m2: [sign:1][exp:5][mantissa:2], fp16: [sign:1][exp:5][mantissa:10]
// Exponent bias is 15 for both formats, so only mantissa needs zero-extension.
__device__ __forceinline__ void fp8e5m2x4_to_half2x2(
    const uint32_t p, uint32_t& out_lo, uint32_t& out_hi)
{
    // p = [b3:b2:b1:b0], each byte is fp8e5m2.
    // Build byte pairs as [b1:0:b0:0] and [b3:0:b2:0] using V_PERM_B32.
    out_lo = __builtin_amdgcn_perm(0u, p, 0x010c000cu);
    out_hi = __builtin_amdgcn_perm(0u, p, 0x030c020cu);
}

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
          int kRepeatN,
          bool kUseFp8E5M2>
__global__ void scaled_mm_kernel(
    const half* __restrict__ const a,
    const uint8_t* __restrict__ const b_prepacked,
    const bfloat16_t* __restrict__ const scale,
    const bfloat16_t* __restrict__ const bias,
    bfloat16_t* __restrict__ const c,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t stride_am,
    const int64_t stride_cm,
    const int has_scale,
    const int has_bias)
{
    constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
    constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;

    // K0xMxK1 layout for A matrix (no extra LDS padding).
    // Apply row permutation on A store to improve LDS local-read banking while
    // keeping compact LDS footprint and 128-bit accesses.
    // K1 = 8 for fp16: enables vec8 LDS reads (like CK)
    constexpr int kK1 = 8;
    // K0 = 16 / 8 = 2
    constexpr int kK0 = kWmmaK / kK1;
    constexpr int kAStrideK1 = kK1;
    constexpr int kShASize = kUnrollK * kK0 * kBlockM * kAStrideK1;

    // B uses KxN layout for efficient vec16 stores during loading
    // C-shuffle epilogue reuses sh_a and sh_b memory. Each warp needs 16*16 halfs.
    constexpr int kCStride = kWmmaN;  // 16 halfs per row

    union SharedStorage {
        struct {
            half a[kShASize];
            uint8_t b[kUnrollK][kBlockN][kWmmaK];
        } ab;
        bfloat16_t c[kBlockWarpsM * kBlockWarpsN][kWmmaM][kCStride];
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
    const auto a_row_phys_to_logi = [&](const int row_phys) -> int {
        const int tile_base = row_phys & ~15;
        const int local = row_phys & 15;
        return tile_base + a_row_phys_to_logi_16(local);
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
            const half* __restrict__ const a_src = a + a_row * stride_am + a_k;

            half* __restrict__ const sh_a_dst = sh_a_row_ptr(stage, k0, m_phys);
            *reinterpret_cast<uint4*>(sh_a_dst) = *reinterpret_cast<const uint4*>(a_src);
        }
    };

    // Loading B: KxN layout with vec16 fp8->fp16 conversion
    constexpr int kBVecs = kBlockN;
    constexpr int kBVecsPerThread = (kBVecs + kThreads - 1) / kThreads;

    const auto load_b_lds_prepacked = [&](const int stage, const int64_t kk) -> void {
        const int ktile = static_cast<int>(kk / kWmmaK);
        #pragma unroll
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec_idx = tid + v * kThreads;
            if (vec_idx >= kBVecs) continue;

            const int col_local = vec_idx;
            const int col = block_n + col_local;

            const int64_t gidx = ((static_cast<int64_t>(ktile) * N + col) * kWmmaK);
            const uint8_t* __restrict__ const b_src = b_prepacked + gidx;

            uint8_t* __restrict__ const b_dst = &sh.ab.b[stage][col_local][0];
            *reinterpret_cast<uint4*>(b_dst) = *reinterpret_cast<const uint4*>(b_src);
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

        using fp16x16_t = uint16_t __attribute__((ext_vector_type(16)));
        using fp32x8_t = float __attribute__((ext_vector_type(8)));

        const int lane_in_subgroup = lane % 16;

        // Pre-load all B fragments (one per rn tile)
        half all_reg_b[kRepeatN][16];
        #pragma unroll
        for (int rn = 0; rn < kRepeatN; ++rn) {
            const int tile_n = warp_n + rn * kBlockWarpsN;
            const int n_col = tile_n * kWmmaN + lane_in_subgroup;
            const uint4 p = *reinterpret_cast<const uint4*>(&sh.ab.b[stage][n_col][0]);
            const uint32_t p32[4] = {p.x, p.y, p.z, p.w};
            uint32_t h32[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if constexpr (kUseFp8E5M2) {
                    fp8e5m2x4_to_half2x2(p32[j], h32[2 * j], h32[2 * j + 1]);
                } else {
                    fp8e4m3x4_to_half2x2(p32[j], h32[2 * j], h32[2 * j + 1]);
                }
            }
            *reinterpret_cast<uint4*>(&all_reg_b[rn][0]) = *reinterpret_cast<const uint4*>(&h32[0]);
            *reinterpret_cast<uint4*>(&all_reg_b[rn][8]) = *reinterpret_cast<const uint4*>(&h32[4]);
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

    // Prologue: load first chunk into LDS using monolithic loads
    #pragma unroll
    for (int u = 0; u < kUnrollK; ++u) {
        const int64_t k = static_cast<int64_t>(u) * kWmmaK;
        load_a_lds_k0mk1(u, k);
    }
    #pragma unroll
    for (int u = 0; u < kUnrollK; ++u) {
        const int64_t k = static_cast<int64_t>(u) * kWmmaK;
        load_b_lds_prepacked(u, k);
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
                load_b_lds_prepacked(u, k);
            }
            asm volatile("s_setprio 0" ::: "memory");
            __syncthreads();
        }
    }

    // Epilogue: C-Shuffle - write output with coalesced vec8 stores
    // Use LDS to transpose from column-major (WMMA layout) to row-major (coalesced)
    if (wave_id < kBlockWarpsM * kBlockWarpsN) {
        bfloat16_t* __restrict__ const sh_c = sh.c[wave_id][0];

        const float scale_f = has_scale ? static_cast<float>(scale[0]) : 1.0f;
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
                    const int row_logi = subgroup * 8 + acc_idx;
                    const int row_phys = c_row_logi_to_phys_16(row_logi);
                    sh_c[row_phys * kCStride + col] = bfloat16_t(acc[repeat_idx][acc_idx] * scale_f);
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

                bfloat16_t* __restrict__ const out_ptr = c + out_row * stride_cm + out_col;
                bfloat16_t* __restrict__ const h = sh_c + read_row_phys * kCStride + read_col_base;

                if (has_bias) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        h[i] = h[i] + bias[out_col + i];
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

extern "C" bool launch_scaled_mm(
    const half* const a,
    const uint8_t* const b_prepacked,
    const bfloat16_t* const scale,
    const bfloat16_t* const bias,
    bfloat16_t* const c,
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

        const bool use_fp8_e5m2 = (b_dtype == 1);

        if (use_fp8_e5m2) {
            hipLaunchKernelGGL(
                (scaled_mm_kernel<kBlockWarpsM, kBlockWarpsN, kUnrollK, kRepeatM, kRepeatN, true>),
                grid, block, 0, stream,
                a, b_prepacked, scale, bias, c,
                M, N, K,
                stride_am, stride_cm,
                has_scale ? 1 : 0, has_bias ? 1 : 0);
        } else {
            hipLaunchKernelGGL(
                (scaled_mm_kernel<kBlockWarpsM, kBlockWarpsN, kUnrollK, kRepeatM, kRepeatN, false>),
                grid, block, 0, stream,
                a, b_prepacked, scale, bias, c,
                M, N, K,
                stride_am, stride_cm,
                has_scale ? 1 : 0, has_bias ? 1 : 0);
        }
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
void scaled_mm(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b_prepacked,
    const std::optional<torch::stable::Tensor>& scale,
    const std::optional<torch::stable::Tensor>& bias,
    torch::stable::Tensor& c,
    const int64_t block_warps_m,
    const int64_t block_warps_n,
    const int64_t unroll_k,
    const int64_t repeat_m,
    const int64_t repeat_n,
    const int64_t b_dtype)
{
    STD_TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    STD_TORCH_CHECK(b_prepacked.is_cuda(), "b_prepacked must be a CUDA tensor");
    STD_TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
    const auto device_index = a.get_device_index();
    STD_TORCH_CHECK(b_prepacked.get_device_index() == device_index, "b_prepacked must be on the same device as a");
    STD_TORCH_CHECK(c.get_device_index() == device_index, "c must be on the same device as a");

    STD_TORCH_CHECK(a.scalar_type() == torch::stable::ScalarType::Half, "a must be float16");
    STD_TORCH_CHECK(c.scalar_type() == torch::stable::ScalarType::BFloat16, "c must be bfloat16");
    STD_TORCH_CHECK(b_dtype == 0 || b_dtype == 1, "b_dtype must be 0 (fp8e4m3) or 1 (fp8e5m2)");
    if (b_dtype == 0) {
        STD_TORCH_CHECK(b_prepacked.scalar_type() == torch::stable::ScalarType::Float8_e4m3fn, "b_prepacked must be float8_e4m3fn when b_dtype=0");
    } else {
        STD_TORCH_CHECK(b_prepacked.scalar_type() == torch::stable::ScalarType::Float8_e5m2, "b_prepacked must be float8_e5m2 when b_dtype=1");
    }

    STD_TORCH_CHECK(a.dim() == 2, "a must be 2D");
    STD_TORCH_CHECK(b_prepacked.dim() == 3, "b_prepacked must be 3D [K/16, N, 16]");
    STD_TORCH_CHECK(c.dim() == 2, "c must be 2D");

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = b_prepacked.size(1);
    STD_TORCH_CHECK(K % kWmmaK == 0, "K must be divisible by 16");
    STD_TORCH_CHECK(b_prepacked.size(0) == K / kWmmaK, "b_prepacked.shape[0] must equal K/16 (", K / kWmmaK, ")");
    STD_TORCH_CHECK(b_prepacked.size(2) == kWmmaK, "b_prepacked.shape[2] must be 16");
    STD_TORCH_CHECK(c.size(0) == M, "c.shape[0] must equal M");
    STD_TORCH_CHECK(c.size(1) == N, "c.shape[1] must equal N");

    // Contiguous fast path requirements
    STD_TORCH_CHECK(a.stride(1) == 1, "a must be row-contiguous (stride(1) == 1)");
    STD_TORCH_CHECK(b_prepacked.stride(2) == 1, "b_prepacked last dim (K=16) must be contiguous");
    STD_TORCH_CHECK(c.stride(1) == 1, "c must be row-contiguous (stride(1) == 1)");

    if (scale.has_value()) {
        STD_TORCH_CHECK(scale.has_value(), "scale must be provided when has_scale=True");
        const auto& scale_t = *scale;
        STD_TORCH_CHECK(scale_t.is_cuda(), "scale must be a CUDA tensor");
        STD_TORCH_CHECK(scale_t.get_device_index() == device_index, "scale must be on the same device as a");
        STD_TORCH_CHECK(scale_t.scalar_type() == torch::stable::ScalarType::BFloat16, "scale must be bfloat16");
        STD_TORCH_CHECK(scale_t.numel() == 1, "scale must have one element");
    }
    if (bias.has_value()) {
        STD_TORCH_CHECK(bias.has_value(), "bias must be provided when has_bias=True");
        const auto& bias_t = *bias;
        STD_TORCH_CHECK(bias_t.is_cuda(), "bias must be a CUDA tensor");
        STD_TORCH_CHECK(bias_t.get_device_index() == device_index, "bias must be on the same device as a");
        STD_TORCH_CHECK(bias_t.scalar_type() == torch::stable::ScalarType::BFloat16, "bias must be bfloat16");
        STD_TORCH_CHECK(bias_t.numel() == N, "bias must have N elements");
    }

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    const half* const a_ptr = reinterpret_cast<const half*>(a.const_data_ptr());
    const uint8_t* const b_ptr = reinterpret_cast<const uint8_t*>(b_prepacked.const_data_ptr());
    const bfloat16_t* const scale_ptr = scale.has_value() ? reinterpret_cast<const bfloat16_t*>(scale->const_data_ptr()) : nullptr;
    const bfloat16_t* const bias_ptr = bias.has_value() ? reinterpret_cast<const bfloat16_t*>(bias->const_data_ptr()) : nullptr;
    bfloat16_t* const c_ptr = reinterpret_cast<bfloat16_t*>(c.mutable_data_ptr());

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

    const bool launched = launch_scaled_mm(
        a_ptr, b_ptr, scale_ptr, bias_ptr, c_ptr,
        M, N, K,
        a.stride(0), c.stride(0),
        scale.has_value() ? 1 : 0, bias.has_value() ? 1 : 0,
        block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n,
        b_dtype, stream);
    STD_TORCH_CHECK(launched, "Unsupported config");

    const hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "scaled_mm kernel launch failed: ", hipGetErrorString(launch_err));
}

STABLE_TORCH_LIBRARY(feather_ops, m)
{
    m.def(
        "scaled_mm("
        "Tensor a, "
        "Tensor b_prepacked, "
        "Tensor? scale, "
        "Tensor? bias, "
        "Tensor(a!) c, "
        "int block_warps_m, "
        "int block_warps_n, "
        "int unroll_k, "
        "int repeat_m, "
        "int repeat_n, "
        "int b_dtype"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_ops, CUDA, m)
{
    m.impl("scaled_mm", TORCH_BOX(&scaled_mm));
}
#endif
