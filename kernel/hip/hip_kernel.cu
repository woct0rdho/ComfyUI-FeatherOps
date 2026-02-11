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

#include <type_traits>

namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
// gfx11 uses wave32 - hardcode for consistent host/device behavior
// rocwmma::Constants::AMDGCN_WAVE_SIZE returns 64 during host compilation
constexpr int kWaveSize = 32;
// K1 = 8 for fp16: enables vec8 LDS reads (like CK)
constexpr int kK1 = 8;
// K0 = kWmmaK / K1 = 16 / 8 = 2
constexpr int kK0 = kWmmaK / kK1;

// -----------------------------------------------------------------------------
// Target contract (torch.compile Tensile kernel on gfx1151).
// This freezes the intended kernel shape/schedule knobs in one place.
//
// Target name tokens:
//   MT128x128x32, WG32_4_1, WS32, PGR1, PLR0, SIA3, 1LDSB1,
//   WSGRA1, WSGRB1, LRVW16
//
// Notes:
// - Only a subset is currently asserted structurally here (tile/workgroup/depthU).
// - Scheduling semantics (PGR1/SIA3/etc.) are implemented incrementally.
// - SU32/SUS256 were explored and not kept in this HIP path.
// -----------------------------------------------------------------------------
struct Gfx1151TorchContract {
    static constexpr int kMacroTileM = 128;      // MT128x...
    static constexpr int kMacroTileN = 128;      // MT...x128x...
    static constexpr int kDepthU = 32;           // MT...x...x32
    static constexpr int kWaveSize = 32;         // WS32
    static constexpr int kWorkgroupWaves = 4;    // WG32_4_1
    static constexpr int kWorkgroupSize = kWaveSize * kWorkgroupWaves; // 128 threads
    static constexpr int kUnrollK = kDepthU / kWmmaK; // 2
    static constexpr int kPrefetchGlobalRead = 1; // PGR1
    static constexpr int kPrefetchLocalRead = 0;  // PLR0
    static constexpr int kScheduleIterAlg = 3;    // SIA3
    static constexpr int kUseOneLdsBuffer = 1;    // 1LDSB1
    static constexpr int kWaveSeparateGRA = 1;    // WSGRA1
    static constexpr int kWaveSeparateGRB = 1;    // WSGRB1
    static constexpr int kLocalReadVectorWidth = 16; // LRVW16
};

__device__ __forceinline__ uint16_t fp8e4m3fn_to_half_bits(uint8_t x)
{
    const uint16_t x_u16 = static_cast<uint16_t>(x);
    const uint16_t sign = static_cast<uint16_t>((x_u16 & 0x80u) << 8);
    uint16_t exp_mant = static_cast<uint16_t>((x_u16 & 0x7Fu) << 7);
    exp_mant = static_cast<uint16_t>(exp_mant + 0x2000u);
    uint16_t bits = static_cast<uint16_t>(sign | exp_mant);
    if ((x_u16 & 0x78u) == 0u) {
        bits = sign;
    }
    return bits;
}

__device__ __forceinline__ half fp8e4m3fn_to_half(uint8_t x)
{
    __half_raw r;
    r.x = fp8e4m3fn_to_half_bits(x);
    return r;
}

// 16-row swizzle used by A LDS physical mapping.
// logical->physical : ((x & 7) << 1) | (x >> 3)
// physical->logical : ((x & 1) << 3) | (x >> 1)
__host__ __device__ __forceinline__ constexpr int a_row_logical_to_phys_16(int x)
{
    return ((x & 7) << 1) | ((x >> 3) & 1);
}

__host__ __device__ __forceinline__ constexpr int a_row_phys_to_logical_16(int x)
{
    return ((x & 1) << 3) | ((x >> 1) & 7);
}

// =============================================================================
// Optimized kernel with K0×M×K1 LDS layout and direct WMMA intrinsics
// K1 = 8 enables vec8 LDS reads (like CK)
// =============================================================================

template <int kBlockWarpsM,
          int kBlockWarpsN,
          int kUnrollK,
          int kStages,
          int kRepeatM,
          int kRepeatN,
          bool kCheckBounds,
          bool kContigFastPath>
__global__ void scaled_mm_kernel_wmma_k0mk1(
    const half* a,
    const uint8_t* b,
    const float* scale,
    const half* bias,
    half* c,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t stride_am,
    int64_t stride_ak,
    int64_t stride_bk,
    int64_t stride_bn,
    int64_t stride_cm,
    int64_t stride_cn,
    int has_scale,
    int has_bias)
{
    // Only wave32 mode supported for gfx11
    // Note: kWaveSize is 64 during host compilation, 32 during gfx11 device compilation
#if __HIP_DEVICE_COMPILE__
    static_assert(kWaveSize == 32, "This kernel requires wave32 mode (gfx11)");
#endif
    static_assert(kStages == 1 || kStages == 2 || kStages == 4, "kStages must be 1, 2, or 4");
    static_assert(kStages >= kUnrollK, "kStages must be >= kUnrollK");
    static_assert(!kContigFastPath || !kCheckBounds,
        "Contiguous fast path requires no-bounds-check launch shape.");

    constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
    constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;
    constexpr bool kMatchesTorchContractShape =
        (kBlockM == Gfx1151TorchContract::kMacroTileM) &&
        (kBlockN == Gfx1151TorchContract::kMacroTileN) &&
        (kUnrollK * kWmmaK == Gfx1151TorchContract::kDepthU) &&
        (kBlockWarpsM == 2) &&
        (kBlockWarpsN == 2) &&
        (kRepeatM == 4) &&
        (kRepeatN == 4);
    // K0×M×K1 layout for A matrix (no extra LDS padding).
    // Apply row permutation on A store to improve LDS local-read banking while
    // keeping compact LDS footprint and 128-bit accesses.
    // B uses K×N layout for efficient vec16 stores during loading
    constexpr int kBPad = 8;
    constexpr int kAStrideK1 = kK1;
    // K0 = kWmmaK / kK1 = 16 / 8 = 2

    // C-shuffle epilogue reuses sh_a memory. Each warp needs 16*24 halfs.
    // Ensure sh_a is large enough for A layout and C-shuffle reuse.
    constexpr int kShASize = kStages * kK0 * kBlockM * kAStrideK1;
    constexpr int kCShuffleSize = kBlockWarpsM * kBlockWarpsN * kWmmaM * (kWmmaN + kBPad);
    static_assert(kShASize >= kCShuffleSize,
        "sh_a too small for C-shuffle epilogue. Increase kStages or kRepeatM.");

    __shared__ __align__(16) half sh_a[kShASize];
    __shared__ __align__(16) half sh_b[kStages][kWmmaK][kBlockN + kBPad];

    const int block_m = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_n = static_cast<int>(blockIdx.x) * kBlockN;

    const int tid = static_cast<int>(threadIdx.x) + static_cast<int>(threadIdx.y) * static_cast<int>(blockDim.x);
    constexpr int kThreads = kWaveSize * kBlockWarpsM * kBlockWarpsN;
    if constexpr (kMatchesTorchContractShape) {
        static_assert(kWaveSize == Gfx1151TorchContract::kWaveSize,
            "Target contract mismatch: expected WS32");
        static_assert(kThreads == Gfx1151TorchContract::kWorkgroupSize,
            "Target contract mismatch: expected WG32_4_1 (128 threads)");
        static_assert(kUnrollK == Gfx1151TorchContract::kUnrollK,
            "Target contract mismatch: expected DepthU=32 with kWmmaK=16");
    }

    // Flattened wave mapping: use 1D thread blocks to mirror rocBLAS/Tensile WG32_4_1 style.
    const int wave_id = tid / kWaveSize;
    const int warp_m = wave_id % kBlockWarpsM;
    const int warp_n = wave_id / kBlockWarpsM;
    const int lane = tid % kWaveSize;

    // Accumulator registers: 8 floats per WMMA tile in wave32 mode
    constexpr int kRepeatTiles = kRepeatM * kRepeatN;
    float acc[kRepeatTiles][8];
    for (int r = 0; r < kRepeatTiles; ++r) {
        for (int i = 0; i < 8; ++i) {
            acc[r][i] = 0.0f;
        }
    }

    // Loading A: K0×M×K1 layout.
    // Structural parity path:
    // 1) physical LDS row mapping (swizzled)
    // 2) inverse mapping for global logical-row source
    // 3) WSGR (wave-separated global read) A-store ownership on torch-shape contract
    constexpr int kAVecs = kK0 * kBlockM; // 2 * 128 = 256 vec8 loads
    constexpr bool kUseWsgrAStoreOwnership =
        kMatchesTorchContractShape && (Gfx1151TorchContract::kWaveSeparateGRA == 1);
    constexpr int kAOwnerWaves =
        kUseWsgrAStoreOwnership ? kBlockWarpsM : (kBlockWarpsM * kBlockWarpsN);
    constexpr int kAOwnerThreads = kAOwnerWaves * kWaveSize;
    constexpr int kAVecsPerOwnerThread = (kAVecs + kAOwnerThreads - 1) / kAOwnerThreads;
    auto a_row_logical_to_phys = [&](int logical_row) -> int {
        const int tile_base = logical_row & ~15;
        const int local = logical_row & 15;
        return tile_base + a_row_logical_to_phys_16(local);
    };
    auto a_row_phys_to_logical = [&](int physical_row) -> int {
        const int tile_base = physical_row & ~15;
        const int local = physical_row & 15;
        return tile_base + a_row_phys_to_logical_16(local);
    };
    auto sh_a_row_ptr = [&](int stage, int k0, int m) -> half* {
        const int idx = (((stage * kK0 + k0) * kBlockM + m) * kAStrideK1);
        return &sh_a[idx];
    };

    auto load_a_lds_k0mk1 = [&](int stage, int64_t kk) {
        // WSGR ownership: only A-owner waves issue A global->LDS stores.
        if constexpr (kUseWsgrAStoreOwnership) {
            if (wave_id >= kAOwnerWaves) return;
        }

        const int a_owner_tid = kUseWsgrAStoreOwnership ? (wave_id * kWaveSize + lane) : tid;

        // Physical LDS space is traversed directly; global logical row is obtained by inverse map.
        for (int v = 0; v < kAVecsPerOwnerThread; ++v) {
            const int vec_idx = a_owner_tid + v * kAOwnerThreads;
            if (vec_idx >= kAVecs) continue;

            // Decode vec_idx to [k0][m_phys].
            const int k0 = vec_idx / kBlockM;
            const int m_phys = vec_idx % kBlockM;
            const int m_logical = a_row_phys_to_logical(m_phys);

            const int64_t a_row = block_m + m_logical;
            const int64_t a_k = kk + k0 * kK1; // Start K position for this K0 slice
            half* sh_a_dst = sh_a_row_ptr(stage, k0, m_phys);
            auto store_a_vec8 = [&](const uint4& packed) {
                *reinterpret_cast<uint4*>(&sh_a_dst[0]) = packed;
            };

            if constexpr (kContigFastPath) {
                const half* a_ptr = a + a_row * stride_am + a_k;
                const uint4 packed = *reinterpret_cast<const uint4*>(a_ptr);
                store_a_vec8(packed);
            } else {
                bool row_in = true;
                if constexpr (kCheckBounds) {
                    row_in = (a_row < M);
                }

                const half* a_ptr = row_in ? (a + a_row * stride_am + a_k * stride_ak) : nullptr;

                bool in_bounds = row_in;
                if constexpr (kCheckBounds) {
                    in_bounds = row_in && (a_k + kK1 - 1 < K);
                }

                // vec8 load (16 bytes)
                const bool can_vec = in_bounds && (stride_ak == 1) &&
                    ((reinterpret_cast<uintptr_t>(a_ptr) & 0xFu) == 0u);

                if (can_vec) {
                    const uint4 packed = *reinterpret_cast<const uint4*>(a_ptr);
                    store_a_vec8(packed);
                } else {
                    // Scalar fallback
                    for (int i = 0; i < kK1; ++i) {
                        half val = __float2half_rn(0.0f);
                        if (row_in) {
                            if constexpr (kCheckBounds) {
                                if (a_k + i < K) {
                                    val = a_ptr[i * stride_ak];
                                }
                            } else {
                                val = a_ptr[i * stride_ak];
                            }
                        }
                        sh_a_dst[i] = val;
                    }
                }
            }
        }
    };

    // Loading B: K×N layout with vec16 fp8→fp16 conversion
    constexpr int kBElements = kWmmaK * kBlockN;
    constexpr int kBVecs = kBElements / 16; // vec16 fp8 loads (16 bytes)
    constexpr int kBVecsPerThread = (kBVecs + kThreads - 1) / kThreads;

    auto load_b_lds = [&](int stage, int64_t kk) {
        // Load B with vec16 fp8→fp16 conversion, store to K×N layout
        for (int v = 0; v < kBVecsPerThread; ++v) {
            const int vec_idx = tid + v * kThreads;
            const int elem_base = vec_idx * 16;
            if (elem_base >= kBElements) continue;

            const int row = elem_base / kBlockN;
            const int col = elem_base % kBlockN;

            const int64_t b_row = kk + row;
            const int64_t b_col = block_n + col;

            if constexpr (kContigFastPath) {
                const uint8_t* b_ptr = b + b_row * stride_bk + b_col;
                const uint4 packed = *reinterpret_cast<const uint4*>(b_ptr);
                const uint32_t* p32 = reinterpret_cast<const uint32_t*>(&packed);
                half h[16];
                for (int j = 0; j < 4; ++j) {
                    uint32_t p = p32[j];
                    h[4 * j + 0] = fp8e4m3fn_to_half(static_cast<uint8_t>(p & 0xFFu));
                    h[4 * j + 1] = fp8e4m3fn_to_half(static_cast<uint8_t>((p >> 8) & 0xFFu));
                    h[4 * j + 2] = fp8e4m3fn_to_half(static_cast<uint8_t>((p >> 16) & 0xFFu));
                    h[4 * j + 3] = fp8e4m3fn_to_half(static_cast<uint8_t>((p >> 24) & 0xFFu));
                }
                uint4* dst_ptr = reinterpret_cast<uint4*>(&sh_b[stage][row][col]);
                dst_ptr[0] = *reinterpret_cast<uint4*>(&h[0]);
                dst_ptr[1] = *reinterpret_cast<uint4*>(&h[8]);
            } else {
                bool row_in = true;
                if constexpr (kCheckBounds) {
                    row_in = (b_row < K);
                }
                const uint8_t* b_ptr = row_in ? (b + b_row * stride_bk + b_col * stride_bn) : nullptr;

                bool in_bounds = row_in && (col + 15 < kBlockN);
                if constexpr (kCheckBounds) {
                    in_bounds = in_bounds && (b_col + 15 < N);
                }
                const bool can_vec = in_bounds && (stride_bn == 1) &&
                    ((reinterpret_cast<uintptr_t>(b_ptr) & 0xFu) == 0u);

                if (can_vec) {
                    const uint4 packed = *reinterpret_cast<const uint4*>(b_ptr);
                    const uint32_t* p32 = reinterpret_cast<const uint32_t*>(&packed);
                    half h[16];
                    for (int j = 0; j < 4; ++j) {
                        uint32_t p = p32[j];
                        h[4 * j + 0] = fp8e4m3fn_to_half(static_cast<uint8_t>(p & 0xFFu));
                        h[4 * j + 1] = fp8e4m3fn_to_half(static_cast<uint8_t>((p >> 8) & 0xFFu));
                        h[4 * j + 2] = fp8e4m3fn_to_half(static_cast<uint8_t>((p >> 16) & 0xFFu));
                        h[4 * j + 3] = fp8e4m3fn_to_half(static_cast<uint8_t>((p >> 24) & 0xFFu));
                    }
                    uint4* dst_ptr = reinterpret_cast<uint4*>(&sh_b[stage][row][col]);
                    dst_ptr[0] = *reinterpret_cast<uint4*>(&h[0]);
                    dst_ptr[1] = *reinterpret_cast<uint4*>(&h[8]);
                } else {
                    for (int i = 0; i < 16; ++i) {
                        if (col + i < kBlockN) {
                            half h = __float2half_rn(0.0f);
                            if (row_in) {
                                if constexpr (kCheckBounds) {
                                    if (b_col + i < N) {
                                        h = fp8e4m3fn_to_half(b_ptr[i * stride_bn]);
                                    }
                                } else {
                                    h = fp8e4m3fn_to_half(b_ptr[i * stride_bn]);
                                }
                            }
                            sh_b[stage][row][col + i] = h;
                        }
                    }
                }
            }
        }
    };

    // Pipeline setup
    constexpr bool kSupportsOverlap = (kStages >= 2 * kUnrollK);
    constexpr bool do_overlap = kSupportsOverlap;
    constexpr int kChunkK = kWmmaK * kUnrollK;
    const int total_chunks = static_cast<int>((K + kChunkK - 1) / kChunkK);

    auto chunk_k0 = [&](int iter_idx) -> int64_t {
        return static_cast<int64_t>(iter_idx) * kChunkK;
    };

    int stage_base = 0;

    const int64_t k0_first = chunk_k0(0);
    int valid_u0 = kUnrollK;
    if constexpr (kCheckBounds) {
        int64_t rem = K - k0_first;
        valid_u0 = rem > 0 ? static_cast<int>((rem + kWmmaK - 1) / kWmmaK) : 0;
        if (valid_u0 > kUnrollK) valid_u0 = kUnrollK;
    }
    for (int u = 0; u < valid_u0; ++u) {
        const int64_t k = k0_first + static_cast<int64_t>(u) * kWmmaK;
        load_a_lds_k0mk1(u, k);
    }
    for (int u = 0; u < valid_u0; ++u) {
        const int64_t k = k0_first + static_cast<int64_t>(u) * kWmmaK;
        load_b_lds(u, k);
    }
    __syncthreads();

    // Main loop
    for (int iter_idx = 0; iter_idx < total_chunks; ++iter_idx) {
        const int64_t k0_iter = chunk_k0(iter_idx);
        const bool has_next = (iter_idx + 1 < total_chunks);
        const int64_t k_next = has_next ? chunk_k0(iter_idx + 1) : 0;

        if constexpr (do_overlap) {
            if (has_next) {
                int valid_u_next = kUnrollK;
                if constexpr (kCheckBounds) {
                    int64_t rem = K - k_next;
                    valid_u_next = rem > 0 ? static_cast<int>((rem + kWmmaK - 1) / kWmmaK) : 0;
                    if (valid_u_next > kUnrollK) valid_u_next = kUnrollK;
                }
                for (int u = 0; u < valid_u_next; ++u) {
                    int stage = 0;
                    if constexpr (kStages == kUnrollK) {
                        stage = u;
                    } else {
                        stage = (stage_base + kUnrollK + u) % kStages;
                    }
                    const int64_t k = k_next + static_cast<int64_t>(u) * kWmmaK;
                    load_a_lds_k0mk1(stage, k);
                }
                for (int u = 0; u < valid_u_next; ++u) {
                    int stage = 0;
                    if constexpr (kStages == kUnrollK) {
                        stage = u;
                    } else {
                        stage = (stage_base + kUnrollK + u) % kStages;
                    }
                    const int64_t k = k_next + static_cast<int64_t>(u) * kWmmaK;
                    load_b_lds(stage, k);
                }
            }
        }

        // Compute: use direct WMMA intrinsic
        if (wave_id < kBlockWarpsM * kBlockWarpsN) {
            int valid_u = kUnrollK;
            if constexpr (kCheckBounds) {
                int64_t rem = K - k0_iter;
                valid_u = rem > 0 ? static_cast<int>((rem + kWmmaK - 1) / kWmmaK) : 0;
                if (valid_u > kUnrollK) valid_u = kUnrollK;
            }
            for (int u = 0; u < valid_u; ++u) {
                int stage = 0;
                if constexpr (kStages == kUnrollK) {
                    stage = u;
                } else {
                    stage = (stage_base + u) % kStages;
                }

                if constexpr (kMatchesTorchContractShape && !do_overlap) {
                    // Per-U LDS wait before local-read + WMMA sequence.
                    asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
                }
                for (int rm = 0; rm < kRepeatM; ++rm) {
                    for (int rn = 0; rn < kRepeatN; ++rn) {
                        const int repeat_idx = rm * kRepeatN + rn;
                        const int tile_m = warp_m + rm * kBlockWarpsM;
                        const int tile_n = warp_n + rn * kBlockWarpsN;

                        // Load A from K0×M×K1 layout into registers.
                        const int lane_in_subgroup = lane % 16;

                        const int m_row = tile_m * kWmmaM + lane_in_subgroup;
                        half reg_a[16];
                        for (int k0 = 0; k0 < kK0; ++k0) {
                            const half* sh_a_src = sh_a_row_ptr(stage, k0, m_row);
                            const uint4 a_vec = *reinterpret_cast<const uint4*>(&sh_a_src[0]);
                            const half* a_halfs = reinterpret_cast<const half*>(&a_vec);
                            #pragma unroll
                            for (int k1 = 0; k1 < kK1; ++k1) {
                                reg_a[k0 * kK1 + k1] = a_halfs[k1];
                            }
                        }

                        // Load B from K×N layout
                        _Float16 reg_b[16];
                        const int n_col = tile_n * kWmmaN + lane_in_subgroup;
                        for (int k = 0; k < kWmmaK; ++k) {
                            reg_b[k] = static_cast<_Float16>(sh_b[stage][k][n_col]);
                        }

                        // Execute WMMA intrinsic
                        // Use _Float16 for vector types (required by HIP/clang)
                        using fp16x16_t = _Float16 __attribute__((ext_vector_type(16)));
                        using float8_t = float __attribute__((ext_vector_type(8)));

                        // Convert half registers to _Float16 for WMMA
                        _Float16 reg_a_fp16[16];
                        for (int i = 0; i < 16; ++i) {
                            reg_a_fp16[i] = static_cast<_Float16>(reg_a[i]);
                        }

                        const fp16x16_t a_frag = *reinterpret_cast<const fp16x16_t*>(reg_a_fp16);
                        const fp16x16_t b_frag = *reinterpret_cast<const fp16x16_t*>(reg_b);
                        float8_t c_frag = *reinterpret_cast<float8_t*>(&acc[repeat_idx][0]);

                        c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);

                        *reinterpret_cast<float8_t*>(&acc[repeat_idx][0]) = c_frag;
                    }
                }
            }
        }

        if constexpr (!do_overlap) {
            if (has_next) {
                if constexpr (kMatchesTorchContractShape) {
                    // Selective asm issue hint around no-overlap A/B load slice.
                    asm volatile("s_setprio 1\n\t" ::: "memory");
                }
                int valid_u_next = kUnrollK;
                if constexpr (kCheckBounds) {
                    int64_t rem = K - k_next;
                    valid_u_next = rem > 0 ? static_cast<int>((rem + kWmmaK - 1) / kWmmaK) : 0;
                    if (valid_u_next > kUnrollK) valid_u_next = kUnrollK;
                }
                for (int u = 0; u < valid_u_next; ++u) {
                    int stage = 0;
                    if constexpr (kStages == kUnrollK) {
                        stage = u;
                    } else {
                        stage = (stage_base + u) % kStages;
                    }
                    const int64_t k = k_next + static_cast<int64_t>(u) * kWmmaK;
                    load_a_lds_k0mk1(stage, k);
                }
                for (int u = 0; u < valid_u_next; ++u) {
                    int stage = 0;
                    if constexpr (kStages == kUnrollK) {
                        stage = u;
                    } else {
                        stage = (stage_base + u) % kStages;
                    }
                    const int64_t k = k_next + static_cast<int64_t>(u) * kWmmaK;
                    load_b_lds(stage, k);
                }
                if constexpr (kMatchesTorchContractShape) {
                    asm volatile("s_setprio 0\n\t" ::: "memory");
                }
            }
        }

        if constexpr (kStages != kUnrollK) {
            stage_base = (stage_base + kUnrollK) % kStages;
        }
        __syncthreads();
    }

    // Epilogue: C-Shuffle - write output with coalesced vec8 stores
    // Use LDS to transpose from column-major (WMMA layout) to row-major (coalesced)
    if (wave_id < kBlockWarpsM * kBlockWarpsN) {
        // Reuse sh_a memory for C-shuffle
        // Each warp gets its own 16×24 buffer (24 = 16 + 8 padding for bank conflicts)
        constexpr int kCPad = 8;
        constexpr int kCStride = kWmmaN + kCPad;  // 24 halfs per row
        half* my_sh_c = sh_a + wave_id * kWmmaM * kCStride;

        const half scale_h = has_scale ? __float2half_rn(scale[0]) : __float2half_rn(1.0f);
        const int subgroup = lane / 16;
        const int lane_in_subgroup = lane % 16;

        for (int rm = 0; rm < kRepeatM; ++rm) {
            for (int rn = 0; rn < kRepeatN; ++rn) {
                const int repeat_idx = rm * kRepeatN + rn;
                const int tile_m = warp_m + rm * kBlockWarpsM;
                const int tile_n = warp_n + rn * kBlockWarpsN;
                const int64_t tile_m_base = block_m + tile_m * kWmmaM;
                const int64_t tile_n_base = block_n + tile_n * kWmmaN;

                // Step 1: Write acc to LDS in column-major order (WMMA layout)
                // Each thread writes 8 values to one column
                const int col = lane_in_subgroup;
                for (int acc_idx = 0; acc_idx < 8; ++acc_idx) {
                    const int row = subgroup * 8 + acc_idx;
                    half val = __float2half_rn(acc[repeat_idx][acc_idx]);
                    val = __hmul(val, scale_h);
                    my_sh_c[row * kCStride + col] = val;
                }

                // Wave executes in lockstep (SIMT), so all writes complete before reads
                // No explicit barrier needed within a wave

                // Step 2: Read from LDS in row-major order for coalesced global write
                // 32 threads -> 16 rows, 2 threads per row, each handles 8 columns
                const int read_row = lane / 2;
                const int col_half = lane % 2;  // 0 = cols 0-7, 1 = cols 8-15
                const int read_col_base = col_half * 8;

                const int64_t out_row = tile_m_base + read_row;
                const int64_t out_col = tile_n_base + read_col_base;

                bool row_ok = true;
                bool col_ok = true;
                if constexpr (kCheckBounds) {
                    row_ok = (out_row < M);
                    col_ok = (out_col + 7 < N);
                }

                if (row_ok && col_ok) {
                    // vec8 read from LDS (16 bytes)
                    uint4 data = *reinterpret_cast<uint4*>(&my_sh_c[read_row * kCStride + read_col_base]);
                    half* h = reinterpret_cast<half*>(&data);

                    // Apply bias if enabled
                    if (has_bias) {
                        for (int i = 0; i < 8; ++i) {
                            h[i] = __hadd(h[i], bias[out_col + i]);
                        }
                    }

                    // vec8 write to global memory (coalesced!)
                    if constexpr (kContigFastPath) {
                        half* out_ptr = &c[out_row * stride_cm + out_col];
                        *reinterpret_cast<uint4*>(out_ptr) = data;
                    } else {
                        // Check alignment and stride for vectorized write.
                        half* out_ptr = &c[out_row * stride_cm + out_col * stride_cn];
                        const bool can_vec = (stride_cn == 1) &&
                            ((reinterpret_cast<uintptr_t>(out_ptr) & 0xFu) == 0u);

                        if (can_vec) {
                            *reinterpret_cast<uint4*>(out_ptr) = data;
                        } else {
                            // Scalar fallback for non-contiguous or unaligned output
                            for (int i = 0; i < 8; ++i) {
                                c[out_row * stride_cm + (out_col + i) * stride_cn] = h[i];
                            }
                        }
                    }
                } else if (row_ok) {
                    // Scalar fallback for edge columns
                    for (int i = 0; i < 8; ++i) {
                        if constexpr (kCheckBounds) {
                            if (out_col + i >= N) break;
                        }
                        half val = my_sh_c[read_row * kCStride + read_col_base + i];
                        if (has_bias) {
                            val = __hadd(val, bias[out_col + i]);
                        }
                        c[out_row * stride_cm + (out_col + i) * stride_cn] = val;
                    }
                }
                // Note: if !row_ok, we skip writing (out of bounds)
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
    bool has_scale,
    bool has_bias,
    int64_t block_warps_m,
    int64_t block_warps_n,
    int64_t unroll_k,
    int64_t stages,
    int64_t repeat_m,
    int64_t repeat_n)
{
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be float16");
    TORCH_CHECK(b.scalar_type() == c10::ScalarType::Float8_e4m3fn, "b must be float8_e4m3fn");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "a and b shapes are incompatible");
    TORCH_CHECK(bias.scalar_type() == at::kHalf || !has_bias, "bias must be float16");

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

    const half* a_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(b.data_ptr());
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* scale_ptr = has_scale ? scale.data_ptr<float>() : nullptr;
    const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;
    half* c_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    auto launch = [&](auto tag) {
        constexpr int kBlockWarpsM = decltype(tag)::kBlockWarpsM;
        constexpr int kBlockWarpsN = decltype(tag)::kBlockWarpsN;
        constexpr int kUnrollK = decltype(tag)::kUnrollK;
        constexpr int kStages = decltype(tag)::kStages;
        constexpr int kRepeatM = decltype(tag)::kRepeatM;
        constexpr int kRepeatN = decltype(tag)::kRepeatN;
        constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
        constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;
        constexpr bool kMatchesTorchContractShape =
            (kBlockM == Gfx1151TorchContract::kMacroTileM) &&
            (kBlockN == Gfx1151TorchContract::kMacroTileN) &&
            (kUnrollK * kWmmaK == Gfx1151TorchContract::kDepthU) &&
            (kBlockWarpsM == 2) &&
            (kBlockWarpsN == 2) &&
            (kRepeatM == 4) &&
            (kRepeatN == 4);

        const bool check_bounds =
            (a.size(0) % kBlockM != 0) ||
            (b.size(1) % kBlockN != 0) ||
            (a.size(1) % (kWmmaK * kUnrollK) != 0);

        constexpr int kThreadsPerBlock = kWaveSize * kBlockWarpsM * kBlockWarpsN;
        static_assert(kThreadsPerBlock <= 1024, "Block size exceeds HIP thread-per-block limit");
        if constexpr (kMatchesTorchContractShape) {
            static_assert(kThreadsPerBlock == Gfx1151TorchContract::kWorkgroupSize,
                "Target contract mismatch: expected 128-thread workgroup for MT128x128x32");
        }
        dim3 block(kThreadsPerBlock, 1, 1);
        dim3 grid(
            (static_cast<uint32_t>(b.size(1)) + kBlockN - 1) / kBlockN,
            (static_cast<uint32_t>(a.size(0)) + kBlockM - 1) / kBlockM);

        auto dispatch_kernel = [&](auto kCheckBoundsVal, auto kContigFastPathVal) {
            constexpr bool kCheckBounds = decltype(kCheckBoundsVal)::value;
            constexpr bool kEnableContigFastPath = decltype(kContigFastPathVal)::value;
            hipLaunchKernelGGL(
                (scaled_mm_kernel_wmma_k0mk1<kBlockWarpsM, kBlockWarpsN, kUnrollK, kStages, kRepeatM, kRepeatN, kCheckBounds, kEnableContigFastPath>),
                grid, block, 0, stream.stream(),
                a_ptr, b_ptr, scale_ptr, bias_ptr, c_ptr,
                a.size(0), b.size(1), a.size(1),
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1), has_scale ? 1 : 0, has_bias ? 1 : 0);
        };

        bool enable_contig_fastpath = false;
        if constexpr (kMatchesTorchContractShape) {
            const auto is_aligned_16 = [](const void* p) {
                return (reinterpret_cast<uintptr_t>(p) & 0xFu) == 0u;
            };
            enable_contig_fastpath =
                !check_bounds &&
                (a.stride(1) == 1) &&
                (b.stride(1) == 1) &&
                (c.stride(1) == 1) &&
                is_aligned_16(a_ptr) &&
                is_aligned_16(b_ptr) &&
                is_aligned_16(c_ptr);
        }

        if (check_bounds) {
            dispatch_kernel(std::true_type{}, std::false_type{});
        } else {
            if (enable_contig_fastpath) {
                dispatch_kernel(std::false_type{}, std::true_type{});
            } else {
                dispatch_kernel(std::false_type{}, std::false_type{});
            }
        }
    };

    auto try_launch = [&](auto tag) {
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

    // Keep only the active fixed config; autotune configs are disabled to reduce compile time.
    bool launched =
        try_launch(ConfigTagK0MK1<2, 2, 2, 2, 4, 4>{});

    /*
    Disabled autotune configs:
        try_launch(ConfigTagK0MK1<2, 2, 2, 4, 4, 4>{}) ||
        try_launch(ConfigTagK0MK1<4, 2, 2, 2, 2, 4>{}) ||
        try_launch(ConfigTagK0MK1<2, 4, 2, 2, 4, 2>{}) ||
        try_launch(ConfigTagK0MK1<2, 2, 4, 4, 4, 4>{}) ||
        try_launch(ConfigTagK0MK1<1, 4, 2, 2, 8, 2>{}) ||
        try_launch(ConfigTagK0MK1<1, 4, 4, 4, 8, 2>{}) ||
        try_launch(ConfigTagK0MK1<4, 1, 4, 4, 2, 8>{}) ||
        try_launch(ConfigTagK0MK1<1, 2, 4, 4, 8, 4>{}) ||
        try_launch(ConfigTagK0MK1<2, 1, 4, 4, 4, 8>{}) ||
        try_launch(ConfigTagK0MK1<1, 1, 4, 4, 8, 8>{}) ||
        try_launch(ConfigTagK0MK1<2, 4, 4, 4, 4, 2>{}) ||
        try_launch(ConfigTagK0MK1<4, 2, 4, 4, 2, 4>{}) ||
        try_launch(ConfigTagK0MK1<2, 4, 2, 2, 4, 4>{}) ||
        try_launch(ConfigTagK0MK1<2, 4, 4, 4, 4, 4>{}) ||
        try_launch(ConfigTagK0MK1<1, 4, 4, 4, 8, 4>{}) ||
        try_launch(ConfigTagK0MK1<1, 8, 4, 4, 8, 2>{}) ||
        try_launch(ConfigTagK0MK1<2, 8, 4, 4, 4, 2>{}) ||
        try_launch(ConfigTagK0MK1<2, 4, 4, 4, 2, 2>{}) ||
        try_launch(ConfigTagK0MK1<2, 2, 4, 4, 8, 4>{}) ||
        try_launch(ConfigTagK0MK1<4, 2, 4, 4, 4, 4>{}) ||
        try_launch(ConfigTagK0MK1<2, 2, 4, 4, 4, 8>{}) ||
        try_launch(ConfigTagK0MK1<4, 4, 4, 4, 2, 2>{}) ||
        try_launch(ConfigTagK0MK1<2, 4, 2, 4, 4, 4>{});
    */

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
