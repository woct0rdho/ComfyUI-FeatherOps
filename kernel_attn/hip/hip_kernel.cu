#ifndef NO_PYTORCH
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#endif

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

using half_bits_t = uint16_t;
using fp8e5m2_t = uint8_t;

typedef uint16_t fp16x16_t __attribute__((ext_vector_type(16)));
typedef float fp32x8_t __attribute__((ext_vector_type(8)));
typedef float fp32x16_t __attribute__((ext_vector_type(16)));

#define HALF16(pointer) (reinterpret_cast<const fp16x16_t*>(&(pointer))[0])
#define HALF16W(pointer) (reinterpret_cast<fp16x16_t*>(&(pointer))[0])

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWaveSize = 32;

__device__ __forceinline__ float half_bits_to_fp32(const half_bits_t h)
{
    const half hv = *reinterpret_cast<const half*>(&h);
    return __half2float(hv);
}

__device__ __forceinline__ half_bits_t fp32_to_half_bits(const float f)
{
    const half hv = __float2half(f);
    return *reinterpret_cast<const half_bits_t*>(&hv);
}

__device__ __forceinline__ fp8e5m2_t half_bits_to_fp8e5m2(const half_bits_t h)
{
    const uint16_t sign = (h >> 8) & 0x80u;
    uint16_t exp = (h >> 10) & 0x1fu;
    uint16_t mant = h & 0x03ffu;

    if (exp == 0) {
        return static_cast<fp8e5m2_t>(sign);
    }

    if (exp == 0x1fu) {
        return static_cast<fp8e5m2_t>(sign | 0x7cu | (mant ? 0x01u : 0x00u));
    }

    uint16_t mant2 = mant >> 8;
    const uint16_t rem = mant & 0x00ffu;
    if (rem > 0x80u || (rem == 0x80u && (mant2 & 1u))) {
        ++mant2;
        if (mant2 == 4u) {
            mant2 = 0;
            ++exp;
            if (exp >= 0x1fu) {
                exp = 0x1fu;
            }
        }
    }

    return static_cast<fp8e5m2_t>(sign | (exp << 2) | mant2);
}

__device__ __forceinline__ half_bits_t fp8e5m2_to_half_bits(const fp8e5m2_t x)
{
    return static_cast<half_bits_t>(x) << 8;
}

__device__ __forceinline__ fp16x16_t load_e5m2x16_as_fp16(const fp8e5m2_t* __restrict__ const p)
{
    fp16x16_t out;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = fp8e5m2_to_half_bits(p[i]);
    }
    return out;
}

__global__ void quantize_kv_e5m2_kernel(
    const half_bits_t* __restrict__ const k,
    const half_bits_t* __restrict__ const v,
    fp8e5m2_t* __restrict__ const k_fp8,
    fp8e5m2_t* __restrict__ const v_fp8,
    const int total,
    const int h,
    const int n_kv,
    const int d,
    const int64_t k_stride0,
    const int64_t k_stride1,
    const int64_t k_stride2,
    const int64_t v_stride0,
    const int64_t v_stride1,
    const int64_t v_stride2)
{
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int d_idx = idx % d;
    const int n_idx = (idx / d) % n_kv;
    const int h_idx = (idx / (d * n_kv)) % h;
    const int b_idx = idx / (d * n_kv * h);

    const int64_t k_off = static_cast<int64_t>(b_idx) * k_stride0 + static_cast<int64_t>(h_idx) * k_stride1 + static_cast<int64_t>(n_idx) * k_stride2 + d_idx;
    const int64_t v_off = static_cast<int64_t>(b_idx) * v_stride0 + static_cast<int64_t>(h_idx) * v_stride1 + static_cast<int64_t>(n_idx) * v_stride2 + d_idx;

    k_fp8[idx] = half_bits_to_fp8e5m2(k[k_off]);
    v_fp8[idx] = half_bits_to_fp8e5m2(v[v_off]);
}

// C = A @ B^T, where A is fp16 Q [Br, d] and B is fp8e5m2 K [Bc, d].
template <int N_WAVES>
__device__ void mul_A_BT(
    const half_bits_t* __restrict__ const A,
    const fp8e5m2_t* __restrict__ const B,
    half_bits_t* __restrict__ const C,
    const int lda,
    const int ldb,
    const int ldc,
    const int m,
    const int n,
    const int k,
    const float scale)
{
    fp16x16_t fragA[2];
    fp16x16_t fragB[2];

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
            fp32x8_t fragACC = {0, 0, 0, 0, 0, 0, 0, 0};

            for (int i = 0; i < k; i += kWmmaK * 2) {
                fragA[0] = HALF16((A + (blk_y * lda + i))[wmma_lane * lda]);
                fragB[0] = load_e5m2x16_as_fp16(B + (blk_x + wmma_lane) * ldb + i);

                fragA[1] = HALF16((A + (blk_y * lda + i + kWmmaK))[wmma_lane * lda]);
                fragB[1] = load_e5m2x16_as_fp16(B + (blk_x + wmma_lane) * ldb + i + kWmmaK);

                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[0], fragB[0], fragACC);
                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[1], fragB[1], fragACC);
            }
            fragACC = fragACC * scale;

            for (int ele = 0; ele < 8; ++ele) {
                const int r = ele * 2 + (lane_id / 16);
                (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane] = fp32_to_half_bits(fragACC[ele]);
            }
        }
    }
}

// C += A @ B, where A is fp16 P [Br, Bc] and B is fp8e5m2 V [Bc, d].
template <int N_WAVES>
__device__ void mul_add_A_B(
    const half_bits_t* __restrict__ const A,
    const fp8e5m2_t* __restrict__ const B,
    half_bits_t* __restrict__ const C,
    const int lda,
    const int ldb,
    const int ldc,
    const int m,
    const int n,
    const int k)
{
    fp16x16_t fragA[2];
    fp16x16_t fragB[2];

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
            fp32x8_t fragACC = {0, 0, 0, 0, 0, 0, 0, 0};

            for (int i = 0; i < k; i += kWmmaK * 2) {
                fragA[0] = HALF16((A + (blk_y * lda + i))[wmma_lane * lda]);
                #pragma unroll
                for (int k_idx = 0; k_idx < 16; ++k_idx) {
                    fragB[0][k_idx] = fp8e5m2_to_half_bits(B[(i + k_idx) * ldb + blk_x + wmma_lane]);
                }

                fragA[1] = HALF16((A + (blk_y * lda + i + kWmmaK))[wmma_lane * lda]);
                #pragma unroll
                for (int k_idx = 0; k_idx < 16; ++k_idx) {
                    fragB[1][k_idx] = fp8e5m2_to_half_bits(B[(i + kWmmaK + k_idx) * ldb + blk_x + wmma_lane]);
                }

                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[0], fragB[0], fragACC);
                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[1], fragB[1], fragACC);
            }

            for (int ele = 0; ele < 8; ++ele) {
                const int r = ele * 2 + (lane_id / 16);
                const float old_val = half_bits_to_fp32((C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane]);
                const float new_val = old_val + fragACC[ele];
                (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane] = fp32_to_half_bits(new_val);
            }
        }
    }
}

template <int Br, int Bc, int N_WAVES>
__global__ void __launch_bounds__(kWaveSize * N_WAVES)
fwd_kernel(
    const half_bits_t* __restrict__ const q,
    const fp8e5m2_t* __restrict__ const k,
    const fp8e5m2_t* __restrict__ const v,
    half_bits_t* __restrict__ const o,
    const int Tr,
    const int Tc,
    const int d,
    const int64_t q_stride0,
    const int64_t q_stride1,
    const int64_t q_stride2,
    const int64_t kv_stride0,
    const int64_t kv_stride1,
    const int64_t kv_stride2,
    const float scale)
{
    const int q_offset = blockIdx.x * q_stride0 + blockIdx.y * q_stride1;
    const int kv_offset = blockIdx.x * kv_stride0 + blockIdx.y * kv_stride1;

    const int ld_q = q_stride2;
    const int ld_kv = kv_stride2;

    const int Tr_i = blockIdx.z;
    if (Tr_i >= Tr) return;

    const int tx = threadIdx.x;

    extern __shared__ half_bits_t sram[];
    half_bits_t* __restrict__ const Si = &sram[0];
    half_bits_t* __restrict__ const Oi = &sram[Br * Bc];

    if (tx < Br) {
        #pragma unroll 4
        for (int i = 0; i < d; i += 16) {
            fp16x16_t zero = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            HALF16W(Oi[tx * d + i]) = zero;
        }
    }
    __syncthreads();

    const half_bits_t* __restrict__ const Qi = q + q_offset + (Tr_i * Br) * ld_q;

    float row_max_old = -INFINITY;
    float l_i = 0;

    for (int j = 0; j < Tc; ++j) {
        const fp8e5m2_t* __restrict__ const Kj = k + kv_offset + (j * Bc) * ld_kv;
        const fp8e5m2_t* __restrict__ const Vj = v + kv_offset + (j * Bc) * ld_kv;

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
                for (int j_idx = 0; j_idx < 16; ++j_idx) {
                    val32 = fmaxf(val32, half_bits_to_fp32(val[j_idx]));
                }
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
                for (int j_idx = 0; j_idx < 16; ++j_idx) {
                    val_f32[j_idx] = half_bits_to_fp32(val[j_idx]);
                }

                val_f32 = val_f32 - row_max_new;

                #pragma unroll
                for (int j_idx = 0; j_idx < 16; ++j_idx) {
                    val_f32[j_idx] = exp2f(val_f32[j_idx]);
                }

                #pragma unroll
                for (int j_idx = 0; j_idx < 16; ++j_idx) {
                    row_sum += val_f32[j_idx];
                }

                #pragma unroll
                for (int j_idx = 0; j_idx < 16; ++j_idx) {
                    val[j_idx] = fp32_to_half_bits(val_f32[j_idx]);
                }

                HALF16W(Si[(tx * Bc) + i]) = val;
            }
            l_i = rowmax_diff_exp * l_i + row_sum;

            #pragma unroll 4
            for (int i = 0; i < d; i += 16) {
                fp16x16_t val = HALF16(Oi[(tx * d) + i]);
                fp32x16_t val_f32;
                #pragma unroll
                for (int j_idx = 0; j_idx < 16; ++j_idx) {
                    val_f32[j_idx] = half_bits_to_fp32(val[j_idx]);
                }

                val_f32 *= rowmax_diff_exp;

                #pragma unroll
                for (int j_idx = 0; j_idx < 16; ++j_idx) {
                    val[j_idx] = fp32_to_half_bits(val_f32[j_idx]);
                }

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
            for (int j_idx = 0; j_idx < 16; ++j_idx) {
                val_f32[j_idx] = half_bits_to_fp32(val[j_idx]);
            }

            val_f32 = val_f32 / l_i;

            #pragma unroll
            for (int j_idx = 0; j_idx < 16; ++j_idx) {
                val[j_idx] = fp32_to_half_bits(val_f32[j_idx]);
            }

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

extern "C" bool launch_attn_fp16_fp8kv(
    const half_bits_t* const q,
    const fp8e5m2_t* const k,
    const fp8e5m2_t* const v,
    half_bits_t* const o,
    const int b,
    const int h,
    const int n,
    const int d,
    const int n_kv,
    const int64_t q_stride0,
    const int64_t q_stride1,
    const int64_t q_stride2,
    const int64_t kv_stride0,
    const int64_t kv_stride1,
    const int64_t kv_stride2,
    const float scale,
    const int Br,
    const int Bc,
    const int N_WAVES,
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

        const int sram_sz = kBr * kBc * sizeof(half_bits_t) + kBr * d * sizeof(half_bits_t);

        hipLaunchKernelGGL(
            (fwd_kernel<kBr, kBc, kNW>),
            grid, block, sram_sz, stream,
            q, k, v, o,
            Tr, Tc, d,
            q_stride0, q_stride1, q_stride2,
            kv_stride0, kv_stride1, kv_stride2,
            scale * 1.4426950408889634f);
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
        try_launch(ConfigTag<128, 64, 8>{}) ||
        try_launch(ConfigTag<128, 32, 16>{}) ||
        try_launch(ConfigTag<128, 32, 8>{}) ||
        try_launch(ConfigTag<64, 32, 16>{}) ||
        try_launch(ConfigTag<32, 64, 8>{}) ||
        try_launch(ConfigTag<64, 32, 8>{}) ||
        try_launch(ConfigTag<32, 32, 8>{}) ||
        try_launch(ConfigTag<16, 32, 2>{}) ||
        false;
}

} // namespace

#ifndef NO_PYTORCH
void attn_fp16_fp8kv(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    torch::stable::Tensor& k_fp8,
    torch::stable::Tensor& v_fp8,
    torch::stable::Tensor& o,
    const int64_t Br,
    const int64_t Bc,
    const int64_t N_WAVES)
{
    STD_TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    STD_TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    STD_TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    STD_TORCH_CHECK(k_fp8.is_cuda(), "k_fp8 must be a CUDA tensor");
    STD_TORCH_CHECK(v_fp8.is_cuda(), "v_fp8 must be a CUDA tensor");
    STD_TORCH_CHECK(o.is_cuda(), "o must be a CUDA tensor");

    const auto device_index = q.get_device_index();
    STD_TORCH_CHECK(k.get_device_index() == device_index, "k must be on the same device as q");
    STD_TORCH_CHECK(v.get_device_index() == device_index, "v must be on the same device as q");
    STD_TORCH_CHECK(k_fp8.get_device_index() == device_index, "k_fp8 must be on the same device as q");
    STD_TORCH_CHECK(v_fp8.get_device_index() == device_index, "v_fp8 must be on the same device as q");
    STD_TORCH_CHECK(o.get_device_index() == device_index, "o must be on the same device as q");

    STD_TORCH_CHECK(q.scalar_type() == torch::stable::ScalarType::Half, "q must be float16");
    STD_TORCH_CHECK(k.scalar_type() == torch::stable::ScalarType::Half, "k must be float16");
    STD_TORCH_CHECK(v.scalar_type() == torch::stable::ScalarType::Half, "v must be float16");
    STD_TORCH_CHECK(k_fp8.scalar_type() == torch::stable::ScalarType::Float8_e5m2, "k_fp8 must be float8_e5m2");
    STD_TORCH_CHECK(v_fp8.scalar_type() == torch::stable::ScalarType::Float8_e5m2, "v_fp8 must be float8_e5m2");
    STD_TORCH_CHECK(o.scalar_type() == torch::stable::ScalarType::Half, "o must be float16");

    STD_TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, d]");
    STD_TORCH_CHECK(k.dim() == 4, "k must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(v.dim() == 4, "v must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(k_fp8.dim() == 4, "k_fp8 must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(v_fp8.dim() == 4, "v_fp8 must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(o.dim() == 4, "o must be 4D [B, H, N, d]");

    const int64_t b = q.size(0);
    const int64_t h = q.size(1);
    const int64_t n = q.size(2);
    const int64_t d = q.size(3);
    const int64_t n_kv = k.size(2);

    STD_TORCH_CHECK(k.size(0) == b && v.size(0) == b && k_fp8.size(0) == b && v_fp8.size(0) == b && o.size(0) == b, "Batch size mismatch");
    STD_TORCH_CHECK(k.size(1) == h && v.size(1) == h && k_fp8.size(1) == h && v_fp8.size(1) == h && o.size(1) == h, "Head size mismatch");
    STD_TORCH_CHECK(k.size(3) == d && v.size(3) == d && k_fp8.size(3) == d && v_fp8.size(3) == d && o.size(3) == d, "Head dim mismatch");
    STD_TORCH_CHECK(v.size(2) == n_kv && k_fp8.size(2) == n_kv && v_fp8.size(2) == n_kv, "KV length mismatch");
    STD_TORCH_CHECK(o.size(2) == n, "Output sequence length mismatch");

    STD_TORCH_CHECK(q.stride(3) == 1, "q.stride(3) must be 1");
    STD_TORCH_CHECK(k.stride(3) == 1, "k.stride(3) must be 1");
    STD_TORCH_CHECK(v.stride(3) == 1, "v.stride(3) must be 1");
    STD_TORCH_CHECK(o.stride(0) == q.stride(0) && o.stride(1) == q.stride(1) && o.stride(2) == q.stride(2) && o.stride(3) == 1, "o must have q-compatible contiguous-last-dim strides");
    STD_TORCH_CHECK(k_fp8.stride(0) == h * n_kv * d && k_fp8.stride(1) == n_kv * d && k_fp8.stride(2) == d && k_fp8.stride(3) == 1, "k_fp8 must be contiguous");
    STD_TORCH_CHECK(v_fp8.stride(0) == h * n_kv * d && v_fp8.stride(1) == n_kv * d && v_fp8.stride(2) == d && v_fp8.stride(3) == 1, "v_fp8 must be contiguous");

    STD_TORCH_CHECK(n % Br == 0, "n must be a multiple of Br");
    STD_TORCH_CHECK(n_kv % Bc == 0, "n_kv must be a multiple of Bc");
    STD_TORCH_CHECK(d % 32 == 0, "d must be a multiple of 32");

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    const half_bits_t* const q_ptr = reinterpret_cast<const half_bits_t*>(q.const_data_ptr());
    const half_bits_t* const k_ptr = reinterpret_cast<const half_bits_t*>(k.const_data_ptr());
    const half_bits_t* const v_ptr = reinterpret_cast<const half_bits_t*>(v.const_data_ptr());
    fp8e5m2_t* const k_fp8_ptr = reinterpret_cast<fp8e5m2_t*>(k_fp8.mutable_data_ptr());
    fp8e5m2_t* const v_fp8_ptr = reinterpret_cast<fp8e5m2_t*>(v_fp8.mutable_data_ptr());
    half_bits_t* const o_ptr = reinterpret_cast<half_bits_t*>(o.mutable_data_ptr());

    const int total_kv = static_cast<int>(b * h * n_kv * d);
    const int threads = 256;
    const int blocks = (total_kv + threads - 1) / threads;
    hipLaunchKernelGGL(
        quantize_kv_e5m2_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        k_ptr, v_ptr, k_fp8_ptr, v_fp8_ptr,
        total_kv,
        static_cast<int>(h), static_cast<int>(n_kv), static_cast<int>(d),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2));
    hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "fp8e5m2 quantize kernel launch failed: ", hipGetErrorString(launch_err));

    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const bool launched = launch_attn_fp16_fp8kv(
        q_ptr,
        k_fp8_ptr,
        v_fp8_ptr,
        o_ptr,
        static_cast<int>(b), static_cast<int>(h), static_cast<int>(n), static_cast<int>(d), static_cast<int>(n_kv),
        q.stride(0), q.stride(1), q.stride(2),
        k_fp8.stride(0), k_fp8.stride(1), k_fp8.stride(2),
        scale,
        static_cast<int>(Br), static_cast<int>(Bc), static_cast<int>(N_WAVES),
        stream);

    STD_TORCH_CHECK(launched, "Unsupported config Br=", Br, " Bc=", Bc, " N_WAVES=", N_WAVES);

    launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "attn_fp16_fp8kv kernel launch failed: ", hipGetErrorString(launch_err));
}

void quantize_kv_e5m2(
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    torch::stable::Tensor& k_fp8,
    torch::stable::Tensor& v_fp8)
{
    STD_TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    STD_TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    STD_TORCH_CHECK(k_fp8.is_cuda(), "k_fp8 must be a CUDA tensor");
    STD_TORCH_CHECK(v_fp8.is_cuda(), "v_fp8 must be a CUDA tensor");

    const auto device_index = k.get_device_index();
    STD_TORCH_CHECK(v.get_device_index() == device_index, "v must be on the same device as k");
    STD_TORCH_CHECK(k_fp8.get_device_index() == device_index, "k_fp8 must be on the same device as k");
    STD_TORCH_CHECK(v_fp8.get_device_index() == device_index, "v_fp8 must be on the same device as k");

    STD_TORCH_CHECK(k.scalar_type() == torch::stable::ScalarType::Half, "k must be float16");
    STD_TORCH_CHECK(v.scalar_type() == torch::stable::ScalarType::Half, "v must be float16");
    STD_TORCH_CHECK(k_fp8.scalar_type() == torch::stable::ScalarType::Float8_e5m2, "k_fp8 must be float8_e5m2");
    STD_TORCH_CHECK(v_fp8.scalar_type() == torch::stable::ScalarType::Float8_e5m2, "v_fp8 must be float8_e5m2");

    STD_TORCH_CHECK(k.dim() == 4, "k must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(v.dim() == 4, "v must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(k_fp8.dim() == 4, "k_fp8 must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(v_fp8.dim() == 4, "v_fp8 must be 4D [B, H, Nkv, d]");

    const int64_t b = k.size(0);
    const int64_t h = k.size(1);
    const int64_t n_kv = k.size(2);
    const int64_t d = k.size(3);

    STD_TORCH_CHECK(v.size(0) == b && v.size(1) == h && v.size(2) == n_kv && v.size(3) == d, "v shape mismatch");
    STD_TORCH_CHECK(k_fp8.size(0) == b && k_fp8.size(1) == h && k_fp8.size(2) == n_kv && k_fp8.size(3) == d, "k_fp8 shape mismatch");
    STD_TORCH_CHECK(v_fp8.size(0) == b && v_fp8.size(1) == h && v_fp8.size(2) == n_kv && v_fp8.size(3) == d, "v_fp8 shape mismatch");

    STD_TORCH_CHECK(k.stride(3) == 1, "k.stride(3) must be 1");
    STD_TORCH_CHECK(v.stride(3) == 1, "v.stride(3) must be 1");
    STD_TORCH_CHECK(k_fp8.stride(0) == h * n_kv * d && k_fp8.stride(1) == n_kv * d && k_fp8.stride(2) == d && k_fp8.stride(3) == 1, "k_fp8 must be contiguous");
    STD_TORCH_CHECK(v_fp8.stride(0) == h * n_kv * d && v_fp8.stride(1) == n_kv * d && v_fp8.stride(2) == d && v_fp8.stride(3) == 1, "v_fp8 must be contiguous");

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    const half_bits_t* const k_ptr = reinterpret_cast<const half_bits_t*>(k.const_data_ptr());
    const half_bits_t* const v_ptr = reinterpret_cast<const half_bits_t*>(v.const_data_ptr());
    fp8e5m2_t* const k_fp8_ptr = reinterpret_cast<fp8e5m2_t*>(k_fp8.mutable_data_ptr());
    fp8e5m2_t* const v_fp8_ptr = reinterpret_cast<fp8e5m2_t*>(v_fp8.mutable_data_ptr());

    const int total_kv = static_cast<int>(b * h * n_kv * d);
    const int threads = 256;
    const int blocks = (total_kv + threads - 1) / threads;
    hipLaunchKernelGGL(
        quantize_kv_e5m2_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        k_ptr, v_ptr, k_fp8_ptr, v_fp8_ptr,
        total_kv,
        static_cast<int>(h), static_cast<int>(n_kv), static_cast<int>(d),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2));
    const hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "fp8e5m2 quantize kernel launch failed: ", hipGetErrorString(launch_err));
}

void attn_fp16_fp8kv_prepacked(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k_fp8,
    const torch::stable::Tensor& v_fp8,
    torch::stable::Tensor& o,
    const int64_t Br,
    const int64_t Bc,
    const int64_t N_WAVES)
{
    STD_TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    STD_TORCH_CHECK(k_fp8.is_cuda(), "k_fp8 must be a CUDA tensor");
    STD_TORCH_CHECK(v_fp8.is_cuda(), "v_fp8 must be a CUDA tensor");
    STD_TORCH_CHECK(o.is_cuda(), "o must be a CUDA tensor");

    const auto device_index = q.get_device_index();
    STD_TORCH_CHECK(k_fp8.get_device_index() == device_index, "k_fp8 must be on the same device as q");
    STD_TORCH_CHECK(v_fp8.get_device_index() == device_index, "v_fp8 must be on the same device as q");
    STD_TORCH_CHECK(o.get_device_index() == device_index, "o must be on the same device as q");

    STD_TORCH_CHECK(q.scalar_type() == torch::stable::ScalarType::Half, "q must be float16");
    STD_TORCH_CHECK(k_fp8.scalar_type() == torch::stable::ScalarType::Float8_e5m2, "k_fp8 must be float8_e5m2");
    STD_TORCH_CHECK(v_fp8.scalar_type() == torch::stable::ScalarType::Float8_e5m2, "v_fp8 must be float8_e5m2");
    STD_TORCH_CHECK(o.scalar_type() == torch::stable::ScalarType::Half, "o must be float16");

    STD_TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, d]");
    STD_TORCH_CHECK(k_fp8.dim() == 4, "k_fp8 must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(v_fp8.dim() == 4, "v_fp8 must be 4D [B, H, Nkv, d]");
    STD_TORCH_CHECK(o.dim() == 4, "o must be 4D [B, H, N, d]");

    const int64_t b = q.size(0);
    const int64_t h = q.size(1);
    const int64_t n = q.size(2);
    const int64_t d = q.size(3);
    const int64_t n_kv = k_fp8.size(2);

    STD_TORCH_CHECK(k_fp8.size(0) == b && v_fp8.size(0) == b && o.size(0) == b, "Batch size mismatch");
    STD_TORCH_CHECK(k_fp8.size(1) == h && v_fp8.size(1) == h && o.size(1) == h, "Head size mismatch");
    STD_TORCH_CHECK(k_fp8.size(3) == d && v_fp8.size(3) == d && o.size(3) == d, "Head dim mismatch");
    STD_TORCH_CHECK(v_fp8.size(2) == n_kv, "KV length mismatch");
    STD_TORCH_CHECK(o.size(2) == n, "Output sequence length mismatch");

    STD_TORCH_CHECK(q.stride(3) == 1, "q.stride(3) must be 1");
    STD_TORCH_CHECK(o.stride(0) == q.stride(0) && o.stride(1) == q.stride(1) && o.stride(2) == q.stride(2) && o.stride(3) == 1, "o must have q-compatible contiguous-last-dim strides");
    STD_TORCH_CHECK(k_fp8.stride(0) == h * n_kv * d && k_fp8.stride(1) == n_kv * d && k_fp8.stride(2) == d && k_fp8.stride(3) == 1, "k_fp8 must be contiguous");
    STD_TORCH_CHECK(v_fp8.stride(0) == h * n_kv * d && v_fp8.stride(1) == n_kv * d && v_fp8.stride(2) == d && v_fp8.stride(3) == 1, "v_fp8 must be contiguous");

    STD_TORCH_CHECK(n % Br == 0, "n must be a multiple of Br");
    STD_TORCH_CHECK(n_kv % Bc == 0, "n_kv must be a multiple of Bc");
    STD_TORCH_CHECK(d % 32 == 0, "d must be a multiple of 32");

    torch::stable::accelerator::DeviceGuard device_guard(device_index);

    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);

    const half_bits_t* const q_ptr = reinterpret_cast<const half_bits_t*>(q.const_data_ptr());
    const fp8e5m2_t* const k_fp8_ptr = reinterpret_cast<const fp8e5m2_t*>(k_fp8.const_data_ptr());
    const fp8e5m2_t* const v_fp8_ptr = reinterpret_cast<const fp8e5m2_t*>(v_fp8.const_data_ptr());
    half_bits_t* const o_ptr = reinterpret_cast<half_bits_t*>(o.mutable_data_ptr());

    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const bool launched = launch_attn_fp16_fp8kv(
        q_ptr,
        k_fp8_ptr,
        v_fp8_ptr,
        o_ptr,
        static_cast<int>(b), static_cast<int>(h), static_cast<int>(n), static_cast<int>(d), static_cast<int>(n_kv),
        q.stride(0), q.stride(1), q.stride(2),
        k_fp8.stride(0), k_fp8.stride(1), k_fp8.stride(2),
        scale,
        static_cast<int>(Br), static_cast<int>(Bc), static_cast<int>(N_WAVES),
        stream);

    STD_TORCH_CHECK(launched, "Unsupported config Br=", Br, " Bc=", Bc, " N_WAVES=", N_WAVES);

    const hipError_t launch_err = hipGetLastError();
    STD_TORCH_CHECK(launch_err == hipSuccess, "attn_fp16_fp8kv_prepacked kernel launch failed: ", hipGetErrorString(launch_err));
}

STABLE_TORCH_LIBRARY(feather_attn_fp16, m)
{
    m.def(
        "attn_fp16_fp8kv("
        "Tensor q, "
        "Tensor k, "
        "Tensor v, "
        "Tensor(b!) k_fp8, "
        "Tensor(c!) v_fp8, "
        "Tensor(a!) o, "
        "int Br, "
        "int Bc, "
        "int N_WAVES"
        ") -> ()");
    m.def(
        "quantize_kv_e5m2("
        "Tensor k, "
        "Tensor v, "
        "Tensor(a!) k_fp8, "
        "Tensor(b!) v_fp8"
        ") -> ()");
    m.def(
        "attn_fp16_fp8kv_prepacked("
        "Tensor q, "
        "Tensor k_fp8, "
        "Tensor v_fp8, "
        "Tensor(a!) o, "
        "int Br, "
        "int Bc, "
        "int N_WAVES"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn_fp16, CUDA, m)
{
    m.impl("attn_fp16_fp8kv", TORCH_BOX(&attn_fp16_fp8kv));
    m.impl("quantize_kv_e5m2", TORCH_BOX(&quantize_kv_e5m2));
    m.impl("attn_fp16_fp8kv_prepacked", TORCH_BOX(&attn_fp16_fp8kv_prepacked));
}
#endif
