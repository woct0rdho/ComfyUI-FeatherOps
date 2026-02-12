// rocWMMA requires __half conversions; torch defines HIP no-half macros.
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif
#ifdef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif
#ifdef __HIP_NO_HALF2_OPERATORS__
#undef __HIP_NO_HALF2_OPERATORS__
#endif

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
// gfx11 uses wave32 - hardcode for consistent host/device behavior
// rocwmma::Constants::AMDGCN_WAVE_SIZE returns 64 during host compilation
constexpr int kWaveSize = 32;

// Packed fp8e4m3fn → fp16 conversion: converts 4 fp8 bytes in a uint32 to 4 fp16 values.
// Produces two uint32s in sequential half2 order: out_lo=[h1:h0], out_hi=[h3:h2].
// Ignores denormals (values with zero exponent map to small fp16 instead of zero).
__device__ __forceinline__ void fp8x4_to_half2x2(
    const uint32_t p, uint32_t& out_lo, uint32_t& out_hi)
{
    // p = [b3:b2:b1:b0], each byte is fp8e4m3fn
    // Pair bytes for sequential output: [b1:b0] and [b3:b2]
    // lo_pair has b0 in [7:0] and b1 in [23:16]
    // hi_pair has b2 in [7:0] and b3 in [23:16]
    const uint32_t lo_pair = (p & 0xFFu) | ((p & 0xFF00u) << 8);       // [0:b1:0:b0]
    const uint32_t hi_pair = ((p >> 16) & 0xFFu) | ((p >> 8) & 0xFF0000u); // [0:b3:0:b2]

    // Convert each pair simultaneously using 32-bit ops:
    // For each byte x at [7:0] or [23:16]:
    //   fp16 = ((x & 0x80) << 8) | (((x & 0x7F) << 7) + 0x2000)
    {
        const uint32_t signs = (lo_pair & 0x00800080u) << 8;
        const uint32_t em = ((lo_pair & 0x007F007Fu) << 7) + 0x20002000u;
        out_lo = signs | em;  // [fp16(b1) : fp16(b0)]
    }
    {
        const uint32_t signs = (hi_pair & 0x00800080u) << 8;
        const uint32_t em = ((hi_pair & 0x007F007Fu) << 7) + 0x20002000u;
        out_hi = signs | em;  // [fp16(b3) : fp16(b2)]
    }
}

// 16-row swizzle used by A LDS physical mapping.
__device__ __forceinline__ constexpr int a_row_logical_to_phys_16(const int x)
{
    return ((x & 7) << 1) | ((x >> 3) & 1);
}

__device__ __forceinline__ constexpr int a_row_phys_to_logical_16(const int x)
{
    return ((x & 1) << 3) | ((x >> 1) & 7);
}

// =============================================================================
// Optimized kernel with K0×M×K1 LDS layout and direct WMMA intrinsics
// Contiguous fast path only: requires aligned, contiguous inputs with
// dimensions divisible by tile sizes.
// K1 = 8 enables vec8 LDS reads (like CK)
// =============================================================================

template <int kBlockWarpsM,
          int kBlockWarpsN,
          int kUnrollK,
          int kStages,
          int kRepeatM,
          int kRepeatN>
__global__ void scaled_mm_kernel_wmma_k0mk1(
    const half* const a,
    const uint8_t* const b,
    const float* const scale,
    const half* const bias,
    half* const c,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t stride_am,
    const int64_t stride_bk,
    const int64_t stride_cm,
    const int has_scale,
    const int has_bias)
{
    static_assert(kStages == 1 || kStages == 2 || kStages == 4, "kStages must be 1, 2, or 4");
    static_assert(kStages >= kUnrollK, "kStages must be >= kUnrollK");

    constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
    constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;
    static_assert(kBlockM % 16 == 0, "kBlockM must be a multiple of 16 (required by row swizzle)");
    static_assert(kBlockN % 16 == 0, "kBlockN must be a multiple of 16 (required by vec16 load)");

    // K0×M×K1 layout for A matrix (no extra LDS padding).
    // Apply row permutation on A store to improve LDS local-read banking while
    // keeping compact LDS footprint and 128-bit accesses.
    // K1 = 8 for fp16: enables vec8 LDS reads (like CK)
    constexpr int kK1 = 8;
    // K0 = kWmmaK / K1 = 16 / 8 = 2
    constexpr int kK0 = kWmmaK / kK1;
    constexpr int kAStrideK1 = kK1;
    constexpr int kShASize = kStages * kK0 * kBlockM * kAStrideK1;

    // B uses K×N layout for efficient vec16 stores during loading
    constexpr int kBPad = 8;

    // C-shuffle epilogue reuses sh_a memory. Each warp needs 16*24 halfs.
    // Ensure sh_a is large enough for A layout and C-shuffle reuse.
    constexpr int kCShuffleSize = kBlockWarpsM * kBlockWarpsN * kWmmaM * (kWmmaN + kBPad);
    static_assert(kShASize >= kCShuffleSize,
        "sh_a too small for C-shuffle epilogue. Increase kStages or kRepeatM.");

    __shared__ __align__(16) half sh_a[kShASize];
    __shared__ __align__(16) half sh_b[kStages][kWmmaK][kBlockN + kBPad];

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

    // Loading A: K0×M×K1 layout with physical/inverse row mapping and
    // wave-separated global-read ownership.
    constexpr int kAVecs = kK0 * kBlockM; // 2 * 128 = 256 vec8 loads
    constexpr bool kUseWsgrAStoreOwnership = true;
    constexpr int kAOwnerWaves =
        kUseWsgrAStoreOwnership ? kBlockWarpsM : (kBlockWarpsM * kBlockWarpsN);
    constexpr int kAOwnerThreads = kAOwnerWaves * kWaveSize;
    constexpr int kAVecsPerOwnerThread = (kAVecs + kAOwnerThreads - 1) / kAOwnerThreads;
    const auto a_row_phys_to_logical = [&](const int physical_row) -> int {
        const int tile_base = physical_row & ~15;
        const int local = physical_row & 15;
        return tile_base + a_row_phys_to_logical_16(local);
    };
    const auto sh_a_row_ptr = [&](const int stage, const int k0, const int m) -> half* {
        const int idx = (((stage * kK0 + k0) * kBlockM + m) * kAStrideK1);
        return &sh_a[idx];
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
            const int m_logical = a_row_phys_to_logical(m_phys);

            const int64_t a_row = block_m + m_logical;
            const int64_t a_k = kk + k0 * kK1; // Start K position for this K0 slice
            half* const sh_a_dst = sh_a_row_ptr(stage, k0, m_phys);

            const half* const a_ptr = a + a_row * stride_am + a_k;
            *reinterpret_cast<uint4*>(sh_a_dst) = *reinterpret_cast<const uint4*>(a_ptr);
        }
    };

    // Loading B: K×N layout with vec16 fp8→fp16 conversion
    constexpr int kBElements = kWmmaK * kBlockN;
    constexpr int kBVecs = kBElements / 16; // vec16 fp8 loads (16 bytes)
    constexpr int kBVecsPerThread = (kBVecs + kThreads - 1) / kThreads;

    const auto load_b_lds = [&](const int stage, const int64_t kk) -> void {
        // Load B with vec16 fp8→fp16 conversion, store to K×N layout
        #pragma unroll
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec_idx = tid + v * kThreads;
            const int elem_base = vec_idx * 16;
            if (elem_base >= kBElements) continue;

            const int row = elem_base / kBlockN;
            const int col = elem_base % kBlockN;

            const int64_t b_row = kk + row;
            const int64_t b_col = block_n + col;

            const uint8_t* const b_ptr = b + b_row * stride_bk + b_col;
            const uint32_t* const p32 = reinterpret_cast<const uint32_t*>(b_ptr);
            uint32_t h32[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                fp8x4_to_half2x2(p32[j], h32[2 * j], h32[2 * j + 1]);
            }
            uint4* const dst_ptr = reinterpret_cast<uint4*>(&sh_b[stage][row][col]);
            dst_ptr[0] = *reinterpret_cast<uint4*>(&h32[0]);
            dst_ptr[1] = *reinterpret_cast<uint4*>(&h32[4]);
        }
    };

    // =========================================================================
    // Split-phase load lambdas for VMEM/WMMA overlap (prefetch path)
    // Phase 1: issue global_load → VGPR buffers (VMEM unit, can overlap with WMMA)
    // Phase 2: VALU convert + ds_write from VGPR buffers → LDS (SIMD unit, serial)
    // =========================================================================

    // A prefetch buffers: one uint4 per vec load, per sub-iteration
    uint4 a_prefetch_buf[kUnrollK][kAVecsPerOwnerThread];

    // B prefetch buffers: 4 uint32_t per vec16 load, per sub-iteration
    uint32_t b_prefetch_buf[kUnrollK][kBVecsPerThread][4];

    // Phase 1 for A: issue global_load → VGPR buffer (no LDS write)
    const auto prefetch_a_global = [&](const int u, const int64_t kk) -> void {
        if constexpr (kUseWsgrAStoreOwnership) {
            if (wave_id >= kAOwnerWaves) return;
        }
        const int a_owner_tid = kUseWsgrAStoreOwnership ? (wave_id * kWaveSize + lane) : tid;

        #pragma unroll
        for (int v = 0; v < kAVecsPerOwnerThread; ++v) {
            const int vec_idx = a_owner_tid + v * kAOwnerThreads;
            if (vec_idx >= kAVecs) continue;

            const int k0 = vec_idx / kBlockM;
            const int m_phys = vec_idx % kBlockM;
            const int m_logical = a_row_phys_to_logical(m_phys);
            const int64_t a_row = block_m + m_logical;
            const int64_t a_k = kk + k0 * kK1;

            const half* const a_ptr = a + a_row * stride_am + a_k;
            a_prefetch_buf[u][v] = *reinterpret_cast<const uint4*>(a_ptr);
        }
    };

    // Phase 2 for A: write VGPR buffer → LDS (ds_write only, no global access)
    const auto commit_a_lds = [&](const int u, const int stage) -> void {
        if constexpr (kUseWsgrAStoreOwnership) {
            if (wave_id >= kAOwnerWaves) return;
        }
        const int a_owner_tid = kUseWsgrAStoreOwnership ? (wave_id * kWaveSize + lane) : tid;

        #pragma unroll
        for (int v = 0; v < kAVecsPerOwnerThread; ++v) {
            const int vec_idx = a_owner_tid + v * kAOwnerThreads;
            if (vec_idx >= kAVecs) continue;

            const int k0 = vec_idx / kBlockM;
            const int m_phys = vec_idx % kBlockM;
            half* const sh_a_dst = sh_a_row_ptr(stage, k0, m_phys);
            *reinterpret_cast<uint4*>(sh_a_dst) = a_prefetch_buf[u][v];
        }
    };

    // Phase 1 for B: issue global_load → VGPR buffer (raw fp8 bytes, no conversion)
    const auto prefetch_b_global = [&](const int u, const int64_t kk) -> void {
        #pragma unroll
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec_idx = tid + v * kThreads;
            const int elem_base = vec_idx * 16;
            if (elem_base >= kBElements) continue;

            const int row = elem_base / kBlockN;
            const int col = elem_base % kBlockN;
            const int64_t b_row = kk + row;
            const int64_t b_col = block_n + col;

            const uint8_t* const b_ptr = b + b_row * stride_bk + b_col;
            const uint32_t* const p32 = reinterpret_cast<const uint32_t*>(b_ptr);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                b_prefetch_buf[u][v][j] = p32[j];
            }
        }
    };

    // Phase 2 for B: VALU fp8→fp16 convert + ds_write from VGPR buffer → LDS
    const auto commit_b_lds = [&](const int u, const int stage) -> void {
        #pragma unroll
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec_idx = tid + v * kThreads;
            const int elem_base = vec_idx * 16;
            if (elem_base >= kBElements) continue;

            const int row = elem_base / kBlockN;
            const int col = elem_base % kBlockN;

            uint32_t h32[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                fp8x4_to_half2x2(b_prefetch_buf[u][v][j], h32[2 * j], h32[2 * j + 1]);
            }
            uint4* const dst_ptr = reinterpret_cast<uint4*>(&sh_b[stage][row][col]);
            dst_ptr[0] = *reinterpret_cast<uint4*>(&h32[0]);
            dst_ptr[1] = *reinterpret_cast<uint4*>(&h32[4]);
        }
    };

    // Pipeline setup
    constexpr int kChunkK = kWmmaK * kUnrollK;
    const int total_chunks = static_cast<int>(K / kChunkK);

    const auto chunk_k0 = [&](const int iter_idx) -> int64_t {
        return static_cast<int64_t>(iter_idx) * kChunkK;
    };

    // True double-buffering requires kStages >= 2*kUnrollK (e.g. kStages=4, kUnrollK=2).
    // Read from stages [stage_base .. stage_base+kUnrollK-1],
    // write to stages [(stage_base+kUnrollK) .. (stage_base+2*kUnrollK-1)] % kStages.
    // Only 1 __syncthreads() per iteration since read/write stage sets are disjoint.
    constexpr bool kDoubleBuffer = (kStages >= 2 * kUnrollK);

    // WMMA compute lambda for one sub-iteration (one stage).
    // Register-tiling: load all B fragments once, then iterate rm with A loads.
    // Reduces LDS reads from (kRepeatM*kRepeatN)*(16+16) to kRepeatN*16 + kRepeatM*16.
    const auto wmma_compute_stage = [&](const int stage, const int64_t k0_iter) -> void {
        if (wave_id >= kBlockWarpsM * kBlockWarpsN) return;

        using fp16x16_t = _Float16 __attribute__((ext_vector_type(16)));
        using float8_t = float __attribute__((ext_vector_type(8)));

        const int lane_in_subgroup = lane % 16;

        // Pre-load all B fragments (one per rn tile)
        _Float16 all_reg_b[kRepeatN][16];
        #pragma unroll
        for (int rn = 0; rn < kRepeatN; ++rn) {
            const int tile_n = warp_n + rn * kBlockWarpsN;
            const int n_col = tile_n * kWmmaN + lane_in_subgroup;
            #pragma unroll
            for (int k = 0; k < kWmmaK; ++k) {
                all_reg_b[rn][k] = static_cast<_Float16>(sh_b[stage][k][n_col]);
            }
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

    int stage_base = 0;

    // Prologue: load first chunk into LDS using monolithic loads
    #pragma unroll
    for (int u = 0; u < kUnrollK; ++u) {
        const int64_t k = static_cast<int64_t>(u) * kWmmaK;
        load_a_lds_k0mk1(u, k);
    }
    #pragma unroll
    for (int u = 0; u < kUnrollK; ++u) {
        const int64_t k = static_cast<int64_t>(u) * kWmmaK;
        load_b_lds(u, k);
    }
    __syncthreads();

    // =========================================================================
    // Main loop with VMEM/WMMA overlap via split-phase loads + ASM fences.
    //
    // When kDoubleBuffer (kStages >= 2*kUnrollK):
    //   Read stages [stage_base..+kUnrollK-1], write stages [stage_base+kUnrollK..+2*kUnrollK-1].
    //   Disjoint stage sets → no read/write race → only 1 __syncthreads().
    //
    // Schedule per iteration:
    //   1. prefetch_global(all u) for next chunk → VMEM (fire-and-forget)
    //   2. ASM fence
    //   3. WMMA compute(all u) on current stages → Matrix unit (CONCURRENT with VMEM)
    //   4. s_waitcnt vmcnt(0)
    //   5. commit_lds(all u) to next stages → VALU convert + ds_write
    //   6. __syncthreads() + advance stage_base
    //
    // When !kDoubleBuffer (kStages == kUnrollK):
    //   Fallback: serial compute-then-load with single __syncthreads().
    // =========================================================================
    for (int iter_idx = 0; iter_idx < total_chunks; ++iter_idx) {
        const int64_t k0_iter = chunk_k0(iter_idx);
        const bool has_next = (iter_idx + 1 < total_chunks);
        const int64_t k_next = has_next ? chunk_k0(iter_idx + 1) : 0;

        if constexpr (kDoubleBuffer) {
            // --- Phase 1: Issue global_load for next chunk (VMEM, fire-and-forget) ---
            if (has_next) {
                #pragma unroll
                for (int u = 0; u < kUnrollK; ++u) {
                    const int64_t k = k_next + static_cast<int64_t>(u) * kWmmaK;
                    prefetch_a_global(u, k);
                    prefetch_b_global(u, k);
                }
            }

            // ASM fence: prevent compiler from moving WMMA before global_load
            asm volatile("" ::: "memory");

            // --- Phase 2: WMMA compute on current LDS data (Matrix unit) ---
            // Concurrent with in-flight global_loads on VMEM unit.
            #pragma unroll
            for (int u = 0; u < kUnrollK; ++u) {
                const int read_stage = (stage_base + u) % kStages;
                wmma_compute_stage(read_stage, k0_iter);
            }

            // --- Phase 3: Wait for global loads, then convert+store to NEXT stages ---
            // No barrier needed before commit — write stages are disjoint from read stages.
            if (has_next) {
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                #pragma unroll
                for (int u = 0; u < kUnrollK; ++u) {
                    const int write_stage = (stage_base + kUnrollK + u) % kStages;
                    commit_a_lds(u, write_stage);
                    commit_b_lds(u, write_stage);
                }
            }

            stage_base = (stage_base + kUnrollK) % kStages;
            __syncthreads();
        } else {
            // Fallback: serial compute-then-load (original schedule, 1 sync)
            #pragma unroll
            for (int u = 0; u < kUnrollK; ++u) {
                const int stage = (stage_base + u) % kStages;
                wmma_compute_stage(stage, k0_iter);
            }

            if (has_next) {
                asm volatile("s_setprio 1" ::: "memory");
                #pragma unroll
                for (int u = 0; u < kUnrollK; ++u) {
                    const int stage = (stage_base + u) % kStages;
                    const int64_t k = k_next + static_cast<int64_t>(u) * kWmmaK;
                    load_a_lds_k0mk1(stage, k);
                }
                #pragma unroll
                for (int u = 0; u < kUnrollK; ++u) {
                    const int stage = (stage_base + u) % kStages;
                    const int64_t k = k_next + static_cast<int64_t>(u) * kWmmaK;
                    load_b_lds(stage, k);
                }
                asm volatile("s_setprio 0" ::: "memory");
            }

            if constexpr (kStages != kUnrollK) {
                stage_base = (stage_base + kUnrollK) % kStages;
            }
            __syncthreads();
        }
    }

    // Epilogue: C-Shuffle - write output with coalesced vec8 stores
    // Use LDS to transpose from column-major (WMMA layout) to row-major (coalesced)
    if (wave_id < kBlockWarpsM * kBlockWarpsN) {
        // Reuse sh_a memory for C-shuffle
        // Each warp gets its own 16×24 buffer (24 = 16 + 8 padding for bank conflicts)
        constexpr int kCPad = 8;
        constexpr int kCStride = kWmmaN + kCPad;  // 24 halfs per row
        half* const sh_c = sh_a + wave_id * kWmmaM * kCStride;

        const half scale_h = has_scale ? __float2half_rn(scale[0]) : __float2half_rn(1.0f);
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
                    const int row_logical = subgroup * 8 + acc_idx;
                    const int row_phys = a_row_logical_to_phys_16(row_logical);
                    half val = __float2half_rn(acc[repeat_idx][acc_idx]);
                    val = __hmul(val, scale_h);
                    sh_c[row_phys * kCStride + col] = val;
                }

                // Wave executes in lockstep (SIMT), so all writes complete before reads
                // No explicit barrier needed within a wave

                // Step 2: Read from LDS in row-major order for coalesced global write
                // 32 threads -> 16 rows, 2 threads per row, each handles 8 columns
                const int read_row = lane / 2;
                const int read_row_phys = a_row_logical_to_phys_16(read_row);
                const int col_half = lane % 2;  // 0 = cols 0-7, 1 = cols 8-15
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

// Config tag for K0MK1 kernel (no vec_a/vec_b params - always uses vec8 A, vec16 B)
template <int M, int N, int U, int STAGES, int RM, int RN>
struct ConfigTagK0MK1 {
    static constexpr int kBlockWarpsM = M;
    static constexpr int kBlockWarpsN = N;
    static constexpr int kUnrollK = U;
    static constexpr int kStages = STAGES;
    static constexpr int kRepeatM = RM;
    static constexpr int kRepeatN = RN;
};

torch::Tensor scaled_mm_k0mk1(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale,
    const torch::Tensor& bias,
    const bool has_scale,
    const bool has_bias,
    const int64_t block_warps_m,
    const int64_t block_warps_n,
    const int64_t unroll_k,
    const int64_t stages,
    const int64_t repeat_m,
    const int64_t repeat_n)
{
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be float16");
    TORCH_CHECK(b.scalar_type() == c10::ScalarType::Float8_e4m3fn, "b must be float8_e4m3fn");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "a and b shapes are incompatible");
    TORCH_CHECK(bias.scalar_type() == at::kHalf || !has_bias, "bias must be float16");

    // Contiguous fast path requirements
    TORCH_CHECK(a.stride(1) == 1, "a must be row-contiguous (stride(1) == 1)");
    TORCH_CHECK(b.stride(1) == 1, "b must be row-contiguous (stride(1) == 1)");

    if (has_scale) {
        TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
        TORCH_CHECK(scale.numel() == 1, "scale must have one element");
        TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");
    }
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.numel() == b.size(1), "bias must have N elements");
        TORCH_CHECK(bias.scalar_type() == at::kHalf, "bias must be float16");
    }

    auto c = torch::empty({a.size(0), b.size(1)}, a.options().dtype(at::kHalf));

    const half* const a_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const uint8_t* const b_ptr = reinterpret_cast<const uint8_t*>(b.data_ptr());
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* const scale_ptr = has_scale ? scale.data_ptr<float>() : nullptr;
    const half* const bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;
    half* const c_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    const auto is_aligned_16 = [](const void* const p) {
        return (reinterpret_cast<uintptr_t>(p) & 0xFu) == 0u;
    };
    TORCH_CHECK(is_aligned_16(a_ptr), "a data pointer must be 16-byte aligned");
    TORCH_CHECK(is_aligned_16(b_ptr), "b data pointer must be 16-byte aligned");
    TORCH_CHECK(is_aligned_16(c_ptr), "c data pointer must be 16-byte aligned");

    const auto launch = [&](const auto tag) -> void {
        constexpr int kBlockWarpsM = decltype(tag)::kBlockWarpsM;
        constexpr int kBlockWarpsN = decltype(tag)::kBlockWarpsN;
        constexpr int kUnrollK = decltype(tag)::kUnrollK;
        constexpr int kStages = decltype(tag)::kStages;
        constexpr int kRepeatM = decltype(tag)::kRepeatM;
        constexpr int kRepeatN = decltype(tag)::kRepeatN;
        constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
        constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;

        TORCH_CHECK(a.size(0) % kBlockM == 0,
            "M (", a.size(0), ") must be divisible by kBlockM (", kBlockM, ")");
        TORCH_CHECK(b.size(1) % kBlockN == 0,
            "N (", b.size(1), ") must be divisible by kBlockN (", kBlockN, ")");
        TORCH_CHECK(a.size(1) % (kWmmaK * kUnrollK) == 0,
            "K (", a.size(1), ") must be divisible by kChunkK (", kWmmaK * kUnrollK, ")");

        constexpr int kThreadsPerBlock = kWaveSize * kBlockWarpsM * kBlockWarpsN;
        static_assert(kThreadsPerBlock <= 1024, "Block size exceeds HIP thread-per-block limit");
        const dim3 block(kThreadsPerBlock, 1, 1);
        const dim3 grid(
            static_cast<uint32_t>(b.size(1)) / kBlockN,
            static_cast<uint32_t>(a.size(0)) / kBlockM);

        hipLaunchKernelGGL(
            (scaled_mm_kernel_wmma_k0mk1<kBlockWarpsM, kBlockWarpsN, kUnrollK, kStages, kRepeatM, kRepeatN>),
            grid, block, 0, stream.stream(),
            a_ptr, b_ptr, scale_ptr, bias_ptr, c_ptr,
            a.size(0), b.size(1), a.size(1),
            a.stride(0), b.stride(0), c.stride(0),
            has_scale ? 1 : 0, has_bias ? 1 : 0);
    };

    const auto try_launch = [&](const auto tag) -> bool {
        if (block_warps_m == decltype(tag)::kBlockWarpsM &&
            block_warps_n == decltype(tag)::kBlockWarpsN &&
            unroll_k == decltype(tag)::kUnrollK &&
            stages == decltype(tag)::kStages &&
            repeat_m == decltype(tag)::kRepeatM &&
            repeat_n == decltype(tag)::kRepeatN) {
            launch(tag);
            return true;
        }
        return false;
    };

    // Autotune candidate configs (kept in sync with kernel/hip/hip_kernel.py::_CONFIGS).
    // Format: (warps_m, warps_n, unroll_k, stages, repeat_m, repeat_n)
    const bool launched =
        try_launch(ConfigTagK0MK1<2, 2, 2, 2, 4, 4>{}) ||
        try_launch(ConfigTagK0MK1<2, 4, 2, 2, 4, 2>{}) ||
        try_launch(ConfigTagK0MK1<2, 4, 2, 2, 4, 4>{}) ||
        try_launch(ConfigTagK0MK1<4, 2, 2, 2, 2, 4>{}) ||
        false;

    TORCH_CHECK(launched, "Unsupported K0MK1 config");
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    namespace py = pybind11;
    m.def(
        "scaled_mm_k0mk1",
        &scaled_mm_k0mk1,
        py::arg("a"),
        py::arg("b"),
        py::arg("scale"),
        py::arg("bias"),
        py::arg("has_scale"),
        py::arg("has_bias"),
        py::arg("block_warps_m"),
        py::arg("block_warps_n"),
        py::arg("unroll_k"),
        py::arg("stages"),
        py::arg("repeat_m"),
        py::arg("repeat_n"),
        "Scaled mixed-precision matmul (HIP, K0MK1 layout)");
}
