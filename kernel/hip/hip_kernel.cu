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

#include <limits>
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

enum KernelMode : int {
    kModeFull = 0,
    kModeNoOverlap = 1,
    kModeCommOnly = 2,
    kModeCompOnly = 3,
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
          bool kUseStagger,
          int kMode>
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
    int stagger_u_iters,
    int stagger_stride_k,
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

    constexpr int kBlockM = kWmmaM * kBlockWarpsM * kRepeatM;
    constexpr int kBlockN = kWmmaN * kBlockWarpsN * kRepeatN;
    // K0×M×K1 layout for A matrix: sh_a[stage][K0][M][K1] where K0=2, K1=8
    // This enables vec8 LDS reads for A (8 halfs = 16 bytes)
    // B uses K×N layout for efficient vec16 stores during loading
    constexpr int kBPad = 8;
    // K0 = kWmmaK / kK1 = 16 / 8 = 2

    // C-shuffle epilogue reuses sh_a memory. Each warp needs 16*24 halfs.
    // Ensure sh_a is large enough: kStages * 2 * kBlockM * 8 >= numWarps * 16 * 24
    constexpr int kShASize = kStages * kK0 * kBlockM * kK1;
    constexpr int kCShuffleSize = kBlockWarpsM * kBlockWarpsN * kWmmaM * (kWmmaN + kBPad);
    static_assert(kShASize >= kCShuffleSize,
        "sh_a too small for C-shuffle epilogue. Increase kStages or kRepeatM.");

    __shared__ __align__(16) half sh_a[kStages][kK0][kBlockM][kK1];
    __shared__ __align__(16) half sh_b[kStages][kWmmaK][kBlockN + kBPad];

    const int block_m = static_cast<int>(blockIdx.y) * kBlockM;
    const int block_n = static_cast<int>(blockIdx.x) * kBlockN;

    const int tid = static_cast<int>(threadIdx.x) + static_cast<int>(threadIdx.y) * static_cast<int>(blockDim.x);
    constexpr int kThreads = kWaveSize * kBlockWarpsM * kBlockWarpsN;

    // Flattened wave mapping: use 1D thread blocks to mirror rocBLAS/Tensile WG32_4_1 style.
    const int wave_id = tid / kWaveSize;
    const int warp_m = wave_id % kBlockWarpsM;
    const int warp_n = wave_id / kBlockWarpsM;
    const int lane = tid % kWaveSize;
    constexpr bool mode_no_overlap = (kMode == kModeNoOverlap);
    constexpr bool mode_comm_only = (kMode == kModeCommOnly);
    constexpr bool mode_comp_only = (kMode == kModeCompOnly);

    // Accumulator registers: 8 floats per WMMA tile in wave32 mode
    constexpr int kRepeatTiles = kRepeatM * kRepeatN;
    float acc[kRepeatTiles][8];
    for (int r = 0; r < kRepeatTiles; ++r) {
        for (int i = 0; i < 8; ++i) {
            acc[r][i] = 0.0f;
        }
    }

    // Loading A: K0×M×K1 layout
    // Thread cluster: organize threads to load K0×M elements, each loading K1=8 halfs
    // Total elements per K tile: kK0 * kBlockM * kK1 = 2 * 128 * 8 = 2048 halfs
    // With vec8 loading: 2048 / 8 = 256 vecs, 256 threads → 1 vec per thread
    constexpr int kAVecs = kK0 * kBlockM; // 2 * 128 = 256 vec8 loads
    constexpr int kAVecsPerThread = (kAVecs + kThreads - 1) / kThreads;

    auto load_a_lds_k0mk1 = [&](int stage, int64_t kk) {
        // Load A into K0×M×K1 layout
        // Each thread loads one vec8 (8 halfs = 16 bytes)
        for (int v = 0; v < kAVecsPerThread; ++v) {
            const int vec_idx = tid + v * kThreads;
            if (vec_idx >= kAVecs) continue;

            // Decode vec_idx to [k0][m] position
            const int k0 = vec_idx / kBlockM;
            const int m = vec_idx % kBlockM;

            if constexpr (mode_comp_only) {
                #pragma unroll
                for (int i = 0; i < kK1; ++i) {
                    sh_a[stage][k0][m][i] = __float2half_rn(1.0f);
                }
                continue;
            }

            const int64_t a_row = block_m + m;
            const int64_t a_k = kk + k0 * kK1; // Start K position for this K0 slice

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
                *reinterpret_cast<uint4*>(&sh_a[stage][k0][m][0]) = packed;
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
                    sh_a[stage][k0][m][i] = val;
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

            if constexpr (mode_comp_only) {
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    if (col + i < kBlockN) {
                        sh_b[stage][row][col + i] = __float2half_rn(1.0f);
                    }
                }
                continue;
            }

            const int64_t b_row = kk + row;
            const int64_t b_col = block_n + col;

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
                for(int j=0; j<4; ++j) {
                    uint32_t p = p32[j];
                    h[4*j+0] = fp8e4m3fn_to_half(static_cast<uint8_t>(p & 0xFFu));
                    h[4*j+1] = fp8e4m3fn_to_half(static_cast<uint8_t>((p >> 8) & 0xFFu));
                    h[4*j+2] = fp8e4m3fn_to_half(static_cast<uint8_t>((p >> 16) & 0xFFu));
                    h[4*j+3] = fp8e4m3fn_to_half(static_cast<uint8_t>((p >> 24) & 0xFFu));
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
    };

    // Pipeline setup
    constexpr bool kSupportsOverlap = (kStages >= 2 * kUnrollK);
    constexpr bool do_overlap = kSupportsOverlap && !mode_no_overlap && !mode_comm_only && !mode_comp_only;
    constexpr int kChunkK = kWmmaK * kUnrollK;
    const int total_chunks = static_cast<int>((K + kChunkK - 1) / kChunkK);

    int stagger_chunks = 0;
    if constexpr (kUseStagger) {
        if (stagger_u_iters > 0 && stagger_stride_k > 0 && total_chunks > 0) {
            const int64_t wg_linear = static_cast<int64_t>(blockIdx.y) * static_cast<int64_t>(gridDim.x) +
                                      static_cast<int64_t>(blockIdx.x);
            const int stagger_group = static_cast<int>(wg_linear % static_cast<int64_t>(stagger_u_iters));
            const int64_t stagger_k = static_cast<int64_t>(stagger_group) * static_cast<int64_t>(stagger_stride_k);
            stagger_chunks = static_cast<int>((stagger_k / kChunkK) % total_chunks);
        }
    }

    auto chunk_k0 = [&](int iter_idx) -> int64_t {
        if (total_chunks <= 0) {
            return 0;
        }
        int chunk = iter_idx;
        if constexpr (kUseStagger) {
            chunk += stagger_chunks;
            if (chunk >= total_chunks) {
                chunk %= total_chunks;
            }
        }
        return static_cast<int64_t>(chunk) * kChunkK;
    };

    int stage_base = 0;
    float comm_sink = 0.0f;

    const int64_t k0_first = chunk_k0(0);
    int valid_u0 = kUnrollK;
    if constexpr (kCheckBounds) {
        int64_t rem = K - k0_first;
        valid_u0 = rem > 0 ? static_cast<int>((rem + kWmmaK - 1) / kWmmaK) : 0;
        if (valid_u0 > kUnrollK) valid_u0 = kUnrollK;
    }
    for (int u = 0; u < valid_u0; ++u) {
        const int64_t kk = k0_first + static_cast<int64_t>(u) * kWmmaK;
        load_a_lds_k0mk1(u, kk);
        load_b_lds(u, kk);
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
                    const int64_t kk = k_next + static_cast<int64_t>(u) * kWmmaK;
                    const int stage = (stage_base + kUnrollK + u) % kStages;
                    load_a_lds_k0mk1(stage, kk);
                    load_b_lds(stage, kk);
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
                const int stage = (stage_base + u) % kStages;
                for (int rm = 0; rm < kRepeatM; ++rm) {
                    for (int rn = 0; rn < kRepeatN; ++rn) {
                        const int repeat_idx = rm * kRepeatN + rn;
                        const int tile_m = warp_m + rm * kBlockWarpsM;
                        const int tile_n = warp_n + rn * kBlockWarpsN;

                        // Load A from K0×M×K1 layout into registers
                        // WMMA needs 16 halfs per thread in wave32
                        // A matrix: swizzled access pattern matching CK
                        const int lane_in_subgroup = lane % 16;

                        // CK uses swizzled access: ((lane & 1) << 3) | (lane >> 1)
                        // This maps lane 0-15 to m positions: 0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15
                        // Even lanes get M rows 0-7, odd lanes get M rows 8-15
                        // Both subgroups (lanes 0-15 and 16-31) read the same M rows (duplication)
                        const int swizzled_lane = ((lane_in_subgroup & 1) << 3) | (lane_in_subgroup >> 1);

                        if constexpr (mode_comm_only) {
                            const int tile_m = warp_m + rm * kBlockWarpsM;
                            const int tile_n = warp_n + rn * kBlockWarpsN;
                            const int m_row = tile_m * kWmmaM + swizzled_lane;
                            const int n_col = tile_n * kWmmaN + lane_in_subgroup;
                            comm_sink += __half2float(sh_a[stage][0][m_row][0]);
                            comm_sink += __half2float(sh_b[stage][0][n_col]);
                            continue;
                        }

                        // Load 16 K elements for this lane's A data
                        // From K0×M×K1: [k0][m][k1] where k0=0..1, k1=0..7
                        // m_row uses swizzled_lane directly (0-15), no subgroup offset
                        const int m_row = tile_m * kWmmaM + swizzled_lane;
                        half reg_a[16];
                        for (int k0 = 0; k0 < kK0; ++k0) {
                            // Read vec8 from LDS (this is the key optimization!)
                            const uint4 a_vec = *reinterpret_cast<uint4*>(&sh_a[stage][k0][m_row][0]);
                            const half* a_halfs = reinterpret_cast<const half*>(&a_vec);
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

                        const fp16x16_t a_frag = *reinterpret_cast<fp16x16_t*>(reg_a_fp16);
                        const fp16x16_t b_frag = *reinterpret_cast<fp16x16_t*>(reg_b);
                        float8_t c_frag = *reinterpret_cast<float8_t*>(&acc[repeat_idx][0]);

                        c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);

                        *reinterpret_cast<float8_t*>(&acc[repeat_idx][0]) = c_frag;
                    }
                }
            }
        }

        if constexpr (!do_overlap) {
            if (has_next) {
                __syncthreads();
                int valid_u_next = kUnrollK;
                if constexpr (kCheckBounds) {
                    int64_t rem = K - k_next;
                    valid_u_next = rem > 0 ? static_cast<int>((rem + kWmmaK - 1) / kWmmaK) : 0;
                    if (valid_u_next > kUnrollK) valid_u_next = kUnrollK;
                }
                for (int u = 0; u < valid_u_next; ++u) {
                    const int64_t kk = k_next + static_cast<int64_t>(u) * kWmmaK;
                    const int stage = (stage_base + u) % kStages;
                    load_a_lds_k0mk1(stage, kk);
                    load_b_lds(stage, kk);
                }
            }
        }

        stage_base = (stage_base + kUnrollK) % kStages;
        __syncthreads();
    }

    if constexpr (mode_comm_only) {
        acc[0][0] = comm_sink;
    }

    // Epilogue: C-Shuffle - write output with coalesced vec8 stores
    // Use LDS to transpose from column-major (WMMA layout) to row-major (coalesced)
    if (wave_id < kBlockWarpsM * kBlockWarpsN) {
        // Reuse sh_a memory for C-shuffle
        // Each warp gets its own 16×24 buffer (24 = 16 + 8 padding for bank conflicts)
        constexpr int kCPad = 8;
        constexpr int kCStride = kWmmaN + kCPad;  // 24 halfs per row
        half* my_sh_c = reinterpret_cast<half*>(&sh_a[0][0][0][0]) + wave_id * kWmmaM * kCStride;

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
                    // Check alignment and stride for vectorized write
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
                } else if (row_ok) {
                    // Scalar fallback for edge columns
                    for (int i = 0; i < 8; ++i) {
                        if constexpr (kCheckBounds) {
                            if (out_col + i >= N) break;
                        }
                        half val = my_sh_c[read_row * kCStride + read_col_base + i];
                        if (has_bias) val = __hadd(val, bias[out_col + i]);
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
    int64_t repeat_n,
    int64_t mode,
    int64_t stagger_u_iters,
    int64_t stagger_stride_k)
{
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be float16");
    TORCH_CHECK(b.scalar_type() == c10::ScalarType::Float8_e4m3fn, "b must be float8_e4m3fn");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "a and b shapes are incompatible");
    TORCH_CHECK(bias.scalar_type() == at::kHalf || !has_bias, "bias must be float16");
    TORCH_CHECK(mode >= 0 && mode <= 3, "mode must be in [0, 3]");
    TORCH_CHECK(stagger_u_iters >= 0, "stagger_u_iters must be >= 0");
    TORCH_CHECK(stagger_stride_k > 0, "stagger_stride_k must be > 0");
    TORCH_CHECK(stagger_u_iters <= std::numeric_limits<int>::max(), "stagger_u_iters too large");
    TORCH_CHECK(stagger_stride_k <= std::numeric_limits<int>::max(), "stagger_stride_k too large");

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

        const bool check_bounds =
            (a.size(0) % kBlockM != 0) ||
            (b.size(1) % kBlockN != 0) ||
            (a.size(1) % (kWmmaK * kUnrollK) != 0);

        constexpr int kThreadsPerBlock = kWaveSize * kBlockWarpsM * kBlockWarpsN;
        static_assert(kThreadsPerBlock <= 1024, "Block size exceeds HIP thread-per-block limit");
        dim3 block(kThreadsPerBlock, 1, 1);
        dim3 grid(
            (static_cast<uint32_t>(b.size(1)) + kBlockN - 1) / kBlockN,
            (static_cast<uint32_t>(a.size(0)) + kBlockM - 1) / kBlockM);

        const bool enable_stagger = (stagger_u_iters > 0);

        auto dispatch_kernel = [&](auto kCheckBoundsVal, auto kUseStaggerVal, auto kModeVal) {
            constexpr bool kCheckBounds = decltype(kCheckBoundsVal)::value;
            constexpr bool kEnableStagger = decltype(kUseStaggerVal)::value;
            constexpr int kKernelMode = decltype(kModeVal)::value;
            hipLaunchKernelGGL(
                (scaled_mm_kernel_wmma_k0mk1<kBlockWarpsM, kBlockWarpsN, kUnrollK, kStages, kRepeatM, kRepeatN, kCheckBounds, kEnableStagger, kKernelMode>),
                grid, block, 0, stream.stream(),
                a_ptr, b_ptr, scale_ptr, bias_ptr, c_ptr,
                a.size(0), b.size(1), a.size(1),
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                kEnableStagger ? static_cast<int>(stagger_u_iters) : 0,
                static_cast<int>(stagger_stride_k),
                has_scale ? 1 : 0, has_bias ? 1 : 0);
        };

        auto launch_mode = [&](auto kCheckBoundsVal, auto kUseStaggerVal) {
            switch (mode) {
                case kModeFull:
                    dispatch_kernel(kCheckBoundsVal, kUseStaggerVal, std::integral_constant<int, kModeFull>{});
                    break;
                case kModeNoOverlap:
                    dispatch_kernel(kCheckBoundsVal, kUseStaggerVal, std::integral_constant<int, kModeNoOverlap>{});
                    break;
                case kModeCommOnly:
                    dispatch_kernel(kCheckBoundsVal, kUseStaggerVal, std::integral_constant<int, kModeCommOnly>{});
                    break;
                case kModeCompOnly:
                    dispatch_kernel(kCheckBoundsVal, kUseStaggerVal, std::integral_constant<int, kModeCompOnly>{});
                    break;
                default:
                    TORCH_CHECK(false, "Unsupported mode ", mode);
            }
        };

        if (check_bounds) {
            if (enable_stagger) {
                launch_mode(std::true_type{}, std::true_type{});
            } else {
                launch_mode(std::true_type{}, std::false_type{});
            }
        } else {
            if (enable_stagger) {
                launch_mode(std::false_type{}, std::true_type{});
            } else {
                launch_mode(std::false_type{}, std::false_type{});
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

    bool launched =
        try_launch(ConfigTagK0MK1<2, 2, 2, 4, 4, 4>{}) ||
        try_launch(ConfigTagK0MK1<2, 2, 2, 2, 4, 4>{}) ||
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
        py::arg("mode") = 0,
        py::arg("stagger_u_iters") = 0,
        py::arg("stagger_stride_k") = 128,
        "Scaled mixed-precision matmul (HIP, K0MK1 layout)");
}
