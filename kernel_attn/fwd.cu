#ifndef NO_PYTORCH
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#endif

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cmath>

using bfloat16_t = uint16_t;

typedef uint16_t bf16x16_t __attribute__((ext_vector_type(16)));
typedef float fp32x8_t __attribute__((ext_vector_type(8)));
typedef uint16_t fp16x16_t __attribute__((ext_vector_type(16)));
typedef float fp32x16_t __attribute__((ext_vector_type(16)));

#define HALF16(pointer) (reinterpret_cast<const fp16x16_t*>(&(pointer))[0])
#define FLOAT8(pointer) (reinterpret_cast<const fp32x8_t*>(&(pointer))[0])
#define HALF16W(pointer) (reinterpret_cast<fp16x16_t*>(&(pointer))[0])
#define FLOAT8W(pointer) (reinterpret_cast<fp32x8_t*>(&(pointer))[0])

__device__ __forceinline__ uint16_t fp32_to_bf16(const float f) {
    const uint32_t u = *reinterpret_cast<const uint32_t*>(&f);
    return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ float bf16_to_fp32(uint16_t bf) {
    const uint32_t u = static_cast<uint32_t>(bf) << 16;
    return *reinterpret_cast<const float*>(&u);
}

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWaveSize = 32;

// C = A @ B^T (Q @ K^T)
template <int N_WAVES>
__device__ void mul_A_BT(
    const bfloat16_t* __restrict__ const A, // Q [Br, d]
    const bfloat16_t* __restrict__ const B, // K [Bc, d]
    bfloat16_t* __restrict__ const C, // Si [Br, Bc]
    const int lda, const int ldb, const int ldc,
    const int m, const int n, const int k,
    const float scale)
{
    bf16x16_t fragA[2];
    bf16x16_t fragB[2];

    const int wave_id = threadIdx.x / kWaveSize;
    const int lane_id = threadIdx.x % kWaveSize;
    const int wmma_lane = lane_id % 16;

    const int max_wave = ((m * n) / (kWmmaM * kWmmaN) + N_WAVES - 1) / N_WAVES;
    for (int wave_off = 0; wave_off < max_wave; ++wave_off) {
        const int wave_xy = wave_id + wave_off * N_WAVES;
        const int wave_x = wave_xy % (n / kWmmaN);
        const int wave_y = wave_xy / (n / kWmmaN);

        const int blk_x = wave_x * kWmmaN;
        const int blk_y = wave_y * kWmmaM;

        if ((blk_x < n) && (blk_y < m)) {
            fp32x8_t fragACC = {0,0,0,0,0,0,0,0};

            for (int i = 0; i < k; i += kWmmaK * 2) {
                fragA[0] = HALF16((A + (blk_y * lda + i))[wmma_lane * lda]);
                fragB[0] = HALF16((B + (blk_x * ldb + i))[wmma_lane * ldb]);

                fragA[1] = HALF16((A + (blk_y * lda + i + kWmmaK))[wmma_lane * lda]);
                fragB[1] = HALF16((B + (blk_x * ldb + i + kWmmaK))[wmma_lane * ldb]);

                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[0], fragB[0], fragACC);
                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[1], fragB[1], fragACC);
            }
            fragACC = fragACC * scale;

            for (int ele = 0; ele < 8; ++ele) {
                const int r = ele * 2 + (lane_id / 16);
                (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane] = fp32_to_bf16(fragACC[ele]);
            }
        }
    }
}

// C = P @ V
// P is Si [Br, Bc] (shared memory)
// V is Vj [Bc, d] (global memory)
// C is Oi [Br, d] (shared memory)
template <int N_WAVES>
__device__ void mul_add_A_B(
    const bfloat16_t* __restrict__ const A, // P [Br, Bc]
    const bfloat16_t* __restrict__ const B, // V [Bc, d]
    bfloat16_t* __restrict__ const C, // O [Br, d]
    const int lda, const int ldb, const int ldc,
    const int m, const int n, const int k)
{
    bf16x16_t fragA[2];
    bf16x16_t fragB[2];

    const int wave_id = threadIdx.x / kWaveSize;
    const int lane_id = threadIdx.x % kWaveSize;
    const int wmma_lane = lane_id % 16;

    const int max_wave = ((m * n) / (kWmmaM * kWmmaN) + N_WAVES - 1) / N_WAVES;
    for (int wave_off = 0; wave_off < max_wave; ++wave_off) {
        const int wave_xy = wave_id + wave_off * N_WAVES;
        const int wave_x = wave_xy % (n / kWmmaN);
        const int wave_y = wave_xy / (n / kWmmaN);

        const int blk_x = wave_x * kWmmaN;
        const int blk_y = wave_y * kWmmaM;

        if ((blk_x < n) && (blk_y < m)) {
            fp32x8_t fragACC = {0,0,0,0,0,0,0,0};

            for (int i = 0; i < k; i += kWmmaK * 2) {
                // A is P (row-major), thread holds row.
                fragA[0] = HALF16((A + (blk_y * lda + i))[wmma_lane * lda]);
                // B is V (row-major), thread holds col. Strided read!
                #pragma unroll
                for(int k_idx = 0; k_idx < 16; ++k_idx) {
                    fragB[0][k_idx] = B[(i + k_idx) * ldb + blk_x + wmma_lane];
                }

                fragA[1] = HALF16((A + (blk_y * lda + i + kWmmaK))[wmma_lane * lda]);
                #pragma unroll
                for(int k_idx = 0; k_idx < 16; ++k_idx) {
                    fragB[1][k_idx] = B[(i + kWmmaK + k_idx) * ldb + blk_x + wmma_lane];
                }

                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[0], fragB[0], fragACC);
                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[1], fragB[1], fragACC);
            }

            // Accumulate to C (which is Oi in shared memory)
            for (int ele = 0; ele < 8; ++ele) {
                const int r = ele * 2 + (lane_id / 16);
                const float old_val = bf16_to_fp32((C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane]);
                const float new_val = old_val + fragACC[ele];
                (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane] = fp32_to_bf16(new_val);
            }
        }
    }
}

template <int Br, int Bc, int N_WAVES>
__global__ void __launch_bounds__(kWaveSize * N_WAVES)
fwd_kernel(
    const bfloat16_t* __restrict__ const q,
    const bfloat16_t* __restrict__ const k,
    const bfloat16_t* __restrict__ const v,
    bfloat16_t* __restrict__ const o,
    const int Tr, const int Tc,
    const int d,
    const int64_t q_stride0, const int64_t q_stride1, const int64_t q_stride2,
    const int64_t kv_stride0, const int64_t kv_stride1, const int64_t kv_stride2,
    const float scale)
{
    const int q_offset = blockIdx.x * q_stride0 + blockIdx.y * q_stride1;
    const int kv_offset = blockIdx.x * kv_stride0 + blockIdx.y * kv_stride1;

    const int ld_q = q_stride2;
    const int ld_kv = kv_stride2;

    const int Tr_i = blockIdx.z;
    if (Tr_i >= Tr) return;

    const int tx = threadIdx.x;

    extern __shared__ bfloat16_t sram[];
    bfloat16_t* __restrict__ const Si = &sram[0];       // Br * Bc
    bfloat16_t* __restrict__ const Oi = &sram[Br * Bc]; // Br * d

    if (tx < Br) {
        #pragma unroll 4
        for (int i = 0; i < d; i += 16) {
            fp32x8_t zero = {0,0,0,0,0,0,0,0};
            FLOAT8W(Oi[tx * d + i]) = zero;
        }
    }
    __syncthreads();

    const bfloat16_t* __restrict__ const Qi = q + q_offset + (Tr_i * Br) * ld_q;

    float row_max_old = -INFINITY;
    float l_i = 0;

    for (int j = 0; j < Tc; ++j) {
        const bfloat16_t* __restrict__ const Kj = k + kv_offset + (j * Bc) * ld_kv;
        const bfloat16_t* __restrict__ const Vj = v + kv_offset + (j * Bc) * ld_kv;

        float row_max_new = -INFINITY;
        float row_sum = 0;
        float rowmax_diff_exp = 0;

        mul_A_BT<N_WAVES>(Qi, Kj, Si, ld_q, ld_kv, Bc, Br, Bc, d, scale);
        __syncthreads();

        if (tx < Br) {
            float val32 = row_max_new;
            #pragma unroll 2
            for (int i = 0; i < Bc; i += 16) {
                fp16x16_t val = HALF16(Si[(tx * Bc) + i]);
                #pragma unroll
                for (int j_idx = 0; j_idx < 16; ++j_idx)
                    val32 = fmaxf(val32, bf16_to_fp32((val[j_idx])));
            }
            row_max_new = val32;

            row_max_new = fmaxf(row_max_old, row_max_new);
            rowmax_diff_exp = exp2f(row_max_old - row_max_new);
            row_max_old = row_max_new;

            #pragma unroll 4
            for (int i = 0; i < Bc; i += 16) {
                fp16x16_t val = HALF16(Si[(tx * Bc) + i]);
                fp32x16_t val_f32;
                #pragma unroll
                for (int j_idx = 0; j_idx < 16; ++j_idx)
                    val_f32[j_idx] = bf16_to_fp32((val[j_idx]));

                val_f32 = val_f32 - row_max_new;

                #pragma unroll
                for (int j_idx = 0; j_idx < 16; ++j_idx)
                    val_f32[j_idx] = exp2f(val_f32[j_idx]);

                #pragma unroll
                for (int j_idx = 0; j_idx < 16; ++j_idx)
                    row_sum += val_f32[j_idx];

                #pragma unroll
                for (int j_idx = 0; j_idx < 16; ++j_idx)
                    val[j_idx] = (fp32_to_bf16(val_f32[j_idx]));

                HALF16W(Si[(tx * Bc) + i]) = val;
            }
            l_i = rowmax_diff_exp * l_i + row_sum;

            #pragma unroll 4
            for (int i = 0; i < d; i += 16) {
                fp16x16_t val = HALF16(Oi[(tx * d) + i]);
                fp32x16_t val_f32;
                for(int j_idx = 0; j_idx < 16; ++j_idx)
                    val_f32[j_idx] = bf16_to_fp32((val[j_idx]));

                val_f32 *= rowmax_diff_exp;

                for(int j_idx = 0; j_idx < 16; ++j_idx)
                    val[j_idx] = (fp32_to_bf16(val_f32[j_idx]));

                HALF16W(Oi[(tx * d) + i]) = val;
            }
        }
        __syncthreads();

        mul_add_A_B<N_WAVES>(Si, Vj, Oi, Bc, ld_kv, d, Br, d, Bc);
        __syncthreads();
    }

    if (tx < Br) {
        for (int i = 0; i < d; i += 16) {
            fp16x16_t val = HALF16(Oi[(tx * d) + i]);
            fp32x16_t val_f32;
            #pragma unroll
            for (int j_idx = 0; j_idx < 16; ++j_idx)
                val_f32[j_idx] = bf16_to_fp32((val[j_idx]));

            val_f32 = val_f32 / l_i;

            #pragma unroll
            for (int j_idx = 0; j_idx < 16; ++j_idx)
                val[j_idx] = (fp32_to_bf16(val_f32[j_idx]));

            HALF16W((o + q_offset + (Tr_i * Br) * ld_q)[tx * ld_q + i]) = val;
        }
    }
}

template <int Br, int Bc, int NW>
struct ConfigTag {
    static constexpr int kBr = Br;
    static constexpr int kBc = Bc;
    static constexpr int kN_WAVES = NW;
};

extern "C" bool launch_attn_fwd(
    const bfloat16_t* const q, const bfloat16_t* const k, const bfloat16_t* const v, bfloat16_t* const o,
    const int b, const int h, const int n, const int d, const int n_kv,
    const int64_t q_stride0, const int64_t q_stride1, const int64_t q_stride2,
    const int64_t kv_stride0, const int64_t kv_stride1, const int64_t kv_stride2,
    const float scale,
    const int Br, const int Bc, const int N_WAVES,
    hipStream_t stream)
{
    const auto launch = [&](const auto tag) -> void {
        constexpr int kBr = decltype(tag)::kBr;
        constexpr int kBc = decltype(tag)::kBc;
        constexpr int kNW = decltype(tag)::kN_WAVES;

        const int Tr = n / kBr;
        const int Tc = n_kv / kBc;

        const dim3 block(kWaveSize * kNW);
        const dim3 grid(b, h, Tr);

        const int sram_sz = kBr * kBc * sizeof(bfloat16_t) + kBr * d * sizeof(bfloat16_t);

        hipLaunchKernelGGL(
            (fwd_kernel<kBr, kBc, kNW>),
            grid, block, sram_sz, stream,
            q, k, v, o,
            Tr, Tc, d,
            q_stride0, q_stride1, q_stride2,
            kv_stride0, kv_stride1, kv_stride2,
            scale * 1.4426950408889634f // scale * 1/ln(2)
        );
    };

    const auto try_launch = [&](const auto tag) -> bool {
        if (Br == decltype(tag)::kBr &&
            Bc == decltype(tag)::kBc &&
            N_WAVES == decltype(tag)::kN_WAVES) {
            launch(tag);
            return true;
        }
        return false;
    };

    return
        try_launch(ConfigTag<64, 64, 16>{}) ||
        try_launch(ConfigTag<64, 64, 8>{}) ||
        try_launch(ConfigTag<64, 128, 16>{}) ||
        try_launch(ConfigTag<128, 64, 16>{}) ||
        try_launch(ConfigTag<32, 64, 8>{}) ||
        try_launch(ConfigTag<64, 32, 8>{}) ||
        try_launch(ConfigTag<32, 32, 8>{}) ||
        try_launch(ConfigTag<16, 32, 2>{}) ||
        false;
}

#ifndef NO_PYTORCH
void attn_fwd(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    torch::stable::Tensor& o,
    const int64_t Br,
    const int64_t Bc,
    const int64_t N_WAVES)
{
    STD_TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    STD_TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    STD_TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    STD_TORCH_CHECK(o.is_cuda(), "o must be a CUDA tensor");

    const auto device_index = q.get_device_index();
    STD_TORCH_CHECK(k.get_device_index() == device_index, "k must be on the same device as q");
    STD_TORCH_CHECK(v.get_device_index() == device_index, "v must be on the same device as q");
    STD_TORCH_CHECK(o.get_device_index() == device_index, "o must be on the same device as q");

    STD_TORCH_CHECK(q.scalar_type() == torch::stable::ScalarType::BFloat16, "q must be bfloat16");
    STD_TORCH_CHECK(k.scalar_type() == torch::stable::ScalarType::BFloat16, "k must be bfloat16");
    STD_TORCH_CHECK(v.scalar_type() == torch::stable::ScalarType::BFloat16, "v must be bfloat16");
    STD_TORCH_CHECK(o.scalar_type() == torch::stable::ScalarType::BFloat16, "o must be bfloat16");

    STD_TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, d]");
    STD_TORCH_CHECK(k.dim() == 4, "k must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(v.dim() == 4, "v must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(o.dim() == 4, "o must be 4D [B, H, N, d]");

    const int64_t b = q.size(0);
    const int64_t h = q.size(1);
    const int64_t n = q.size(2);
    const int64_t d = q.size(3);
    const int64_t n_kv = k.size(2);

    STD_TORCH_CHECK(k.size(0) == b && v.size(0) == b && o.size(0) == b, "Batch size mismatch");
    STD_TORCH_CHECK(k.size(1) == h && v.size(1) == h && o.size(1) == h, "Head size mismatch");
    STD_TORCH_CHECK(k.size(3) == d && v.size(3) == d && o.size(3) == d, "Head dim mismatch");
    STD_TORCH_CHECK(v.size(2) == n_kv, "KV length mismatch");

    STD_TORCH_CHECK(n % Br == 0, "n must be a multiple of Br");
    STD_TORCH_CHECK(n_kv % Bc == 0, "n_kv must be a multiple of Bc");
    STD_TORCH_CHECK(d % 32 == 0, "d must be a multiple of 32");

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    const bfloat16_t* const q_ptr = reinterpret_cast<const bfloat16_t*>(q.const_data_ptr());
    const bfloat16_t* const k_ptr = reinterpret_cast<const bfloat16_t*>(k.const_data_ptr());
    const bfloat16_t* const v_ptr = reinterpret_cast<const bfloat16_t*>(v.const_data_ptr());
    bfloat16_t* const o_ptr = reinterpret_cast<bfloat16_t*>(o.mutable_data_ptr());

    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    const bool launched = launch_attn_fwd(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        b, h, n, d, n_kv,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        scale,
        Br, Bc, N_WAVES,
        stream);

    STD_TORCH_CHECK(launched, "Unsupported config Br=", Br, " Bc=", Bc, " N_WAVES=", N_WAVES);

    const hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "attn_fwd kernel launch failed: ", hipGetErrorString(launch_err));
}

STABLE_TORCH_LIBRARY(feather_attn, m)
{
    m.def(
        "attn_fwd("
        "Tensor q, "
        "Tensor k, "
        "Tensor v, "
        "Tensor(a!) o, "
        "int Br, "
        "int Bc, "
        "int N_WAVES"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn, CUDA, m)
{
    m.impl("attn_fwd", TORCH_BOX(&attn_fwd));
}
#endif
