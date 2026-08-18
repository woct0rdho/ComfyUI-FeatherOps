#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "hip_kernel.h"

#include <cstdint>

namespace feather_attn {
namespace {

constexpr int kRowsBlock = 256;
constexpr int kThreads = 128;
constexpr int kQThreads = 512;
constexpr int kWaveSize = 32;
constexpr int kInnerBlock = 16;
constexpr int kHeadDim = 128;
constexpr int kPacked = 8;
constexpr int kTransposeStride = 20;
constexpr int kRowStride = 136;
constexpr int kQOwnerBlock = 256;
constexpr int kKVOwnerBlock = 32;

using Half2 = _Float16 __attribute__((ext_vector_type(2)));
using Half4 = _Float16 __attribute__((ext_vector_type(4)));
using Half8 = _Float16 __attribute__((ext_vector_type(8)));
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using UInt8 = uint32_t __attribute__((ext_vector_type(8)));

__device__ inline int KVLdsOffset(int row, int chunk)
{
    const int physical_chunk = chunk ^ (row % (kHeadDim / kPacked));
    return row * kHeadDim + physical_chunk * kPacked;
}

__device__ inline Half16 LoadKVRow16(
    const _Float16* ptr,
    int row,
    int d_tile)
{
    const int chunk = d_tile * 2;
    const Half8 lo = *reinterpret_cast<const Half8*>(
        ptr + KVLdsOffset(row, chunk));
    const Half8 hi = *reinterpret_cast<const Half8*>(
        ptr + KVLdsOffset(row, chunk + 1));
    Half16 result;
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        result[i] = lo[i];
        result[i + 8] = hi[i];
    }
    return result;
}

__device__ inline Half16 LoadTransposeRow16(
    const _Float16* ptr,
    int row)
{
    Half16 result;
#pragma unroll
    for(int chunk = 0; chunk < kInnerBlock / 4; ++chunk)
    {
        const Half4 value = *reinterpret_cast<const Half4*>(
            ptr + row * kTransposeStride + chunk * 4);
#pragma unroll
        for(int i = 0; i < 4; ++i)
            result[chunk * 4 + i] = value[i];
    }
    return result;
}

__device__ inline void WmmaInPlace(
    Float8& c,
    const Half16& a,
    const Half16& b)
{
    asm volatile(
        "v_wmma_f32_16x16x16_f16 %0, %1, %2, %0"
        : "+v"(c)
        : "v"(a), "v"(b));
}

__device__ inline Half16 CFragmentToA(
    const Float8& fragment,
    int lane)
{
    UInt8 words;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        Half2 pair;
        pair[0] = static_cast<_Float16>(fragment[i * 2]);
        pair[1] = static_cast<_Float16>(fragment[i * 2 + 1]);
        const uint32_t value = __builtin_bit_cast(uint32_t, pair);
        const uint32_t peer = __builtin_amdgcn_permlanex16(
            0u, value, 0x76543210u, 0xfedcba98u, false, true);
        const uint32_t selector_0 =
            lane < 16 ? 0x05040100u : 0x01000504u;
        const uint32_t selector_1 =
            lane < 16 ? 0x07060302u : 0x03020706u;
        words[i * 2] = __builtin_amdgcn_perm(peer, value, selector_0);
        words[i * 2 + 1] = __builtin_amdgcn_perm(
            peer, value, selector_1);
    }
    return __builtin_bit_cast(Half16, words);
}

template<bool kNhd>
__device__ inline int HeadOffset(
    int head_linear,
    int rows,
    int num_heads)
{
    if constexpr(kNhd)
    {
        const int batch = head_linear / num_heads;
        const int head = head_linear - batch * num_heads;
        return (batch * rows * num_heads + head) * kHeadDim;
    }
    return head_linear * rows * kHeadDim;
}

template<bool kNhd>
__device__ inline void DecodeOwner(
    int block,
    int owner_tiles,
    int head_count,
    int num_heads,
    int& owner_tile,
    int& head_linear)
{
    if constexpr(kNhd)
    {
        head_linear = block % head_count;
        owner_tile = block / head_count;
    }
    else
    {
        owner_tile = block % owner_tiles;
        head_linear = block / owner_tiles;
    }
}

template<bool kNhd>
__global__ void DeltaD128Kernel(
    const __half* __restrict__ out,
    const __half* __restrict__ dout,
    float* __restrict__ delta,
    int rows,
    int n_q,
    int num_heads)
{
    const int linear = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear >= rows)
        return;
    int delta_row = linear;
    if constexpr(kNhd)
    {
        const int batch = linear / (n_q * num_heads);
        const int batch_row = linear - batch * n_q * num_heads;
        const int q_row = batch_row / num_heads;
        const int head = batch_row - q_row * num_heads;
        delta_row = (batch * num_heads + head) * n_q + q_row;
    }
    const __half* out_row = out + linear * kHeadDim;
    const __half* dout_row = dout + linear * kHeadDim;
    float value = 0.0f;
#pragma unroll
    for(int d = 0; d < kHeadDim; ++d)
        value += __half2float(out_row[d]) * __half2float(dout_row[d]);
    delta[delta_row] = value;
}

template<bool kNhd, bool kCacheQRows, bool kCacheDoRows>
__global__ __launch_bounds__(kThreads) void D128KVKernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ dout,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    __half* __restrict__ dk,
    __half* __restrict__ dv,
    int n_q,
    int n_kv,
    int num_heads,
    int head_count,
    float scale)
{
    constexpr int v_bytes =
        kKVOwnerBlock * kHeadDim * sizeof(_Float16);
    constexpr int transpose_bytes =
        kHeadDim * kTransposeStride * sizeof(_Float16);
    constexpr int row_bytes =
        kInnerBlock * kRowStride * sizeof(_Float16);
    constexpr int lds_bytes =
        v_bytes + 2 * transpose_bytes +
        (kCacheQRows ? row_bytes : 0) +
        (kCacheDoRows ? row_bytes : 0);
    static_assert(v_bytes == 8192);
    static_assert(transpose_bytes == 5120);
    static_assert(row_bytes == 4352);
    static_assert(lds_bytes >= 18432 && lds_bytes <= 27136);

    alignas(16) __shared__ uint8_t lds[lds_bytes];
    auto* v_lds = reinterpret_cast<_Float16*>(lds);
    auto* q_transpose_lds = reinterpret_cast<_Float16*>(lds + v_bytes);
    auto* do_transpose_lds = reinterpret_cast<_Float16*>(
        lds + v_bytes + transpose_bytes);
    auto* q_row_lds = reinterpret_cast<_Float16*>(
        lds + v_bytes + 2 * transpose_bytes);
    auto* do_row_lds = q_row_lds +
                       (kCacheQRows ? kInnerBlock * kRowStride : 0);

    const int tid = threadIdx.x;
    const int wave = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int lane_row = lane % kInnerBlock;
    const int lane_group = lane / kInnerBlock;
    const bool owns_dk = wave < 2;
    const int owner_wave = wave % 2;
    const int kv_tiles = (n_kv + kKVOwnerBlock - 1) / kKVOwnerBlock;
    int owner_tile;
    int head_linear;
    DecodeOwner<kNhd>(
        static_cast<int>(blockIdx.x),
        kv_tiles,
        head_count,
        num_heads,
        owner_tile,
        head_linear);
    const int kv_start = owner_tile * kKVOwnerBlock;
    const int kv_row = owner_wave * kInnerBlock + lane_row;
    const int q_offset = HeadOffset<kNhd>(head_linear, n_q, num_heads);
    const int kv_offset = HeadOffset<kNhd>(head_linear, n_kv, num_heads);
    const int q_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;
    const int kv_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;

#pragma unroll
    for(int issue = 0; issue < 4; ++issue)
    {
        const int linear_chunk = tid + issue * kThreads;
        const int row = linear_chunk / (kHeadDim / kPacked);
        const int chunk = linear_chunk % (kHeadDim / kPacked);
        Half8 value = {};
        if(kv_start + row < n_kv)
        {
            const int offset = kv_offset +
                               (kv_start + row) * kv_row_stride +
                               chunk * kPacked;
            value = *reinterpret_cast<const Half8*>(v + offset);
        }
        *reinterpret_cast<Half8*>(
            v_lds + KVLdsOffset(row, chunk)) = value;
    }
    __syncthreads();

    Half16 k_rows[kHeadDim / kInnerBlock] = {};
#pragma unroll
    for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
    {
        if(kv_start + kv_row < n_kv)
        {
            const int offset = kv_offset +
                               (kv_start + kv_row) * kv_row_stride +
                               d_tile * kInnerBlock;
            k_rows[d_tile] = *reinterpret_cast<const Half16*>(k + offset);
        }
    }
    Float8 grad_accum[kHeadDim / kInnerBlock] = {};

    for(int q_start = 0; q_start < n_q; q_start += kInnerBlock)
    {
        __syncthreads();
#pragma unroll
        for(int issue = 0; issue < 2; ++issue)
        {
            const int linear_chunk = tid + issue * kThreads;
            const int stage_row = linear_chunk / (kHeadDim / kPacked);
            const int stage_chunk = linear_chunk % (kHeadDim / kPacked);
            Half8 q_stage = {};
            Half8 do_stage = {};
            if(q_start + stage_row < n_q)
            {
                const int offset = q_offset +
                                   (q_start + stage_row) * q_row_stride +
                                   stage_chunk * kPacked;
                q_stage = *reinterpret_cast<const Half8*>(q + offset);
                do_stage = *reinterpret_cast<const Half8*>(dout + offset);
            }
#pragma unroll
            for(int i = 0; i < kPacked; ++i)
            {
                const int d = stage_chunk * kPacked + i;
                q_transpose_lds[d * kTransposeStride + stage_row] =
                    q_stage[i];
                do_transpose_lds[d * kTransposeStride + stage_row] =
                    do_stage[i];
            }
            if constexpr(kCacheQRows)
            {
                *reinterpret_cast<Half8*>(
                    q_row_lds + stage_row * kRowStride +
                    stage_chunk * kPacked) = q_stage;
            }
            if constexpr(kCacheDoRows)
            {
                *reinterpret_cast<Half8*>(
                    do_row_lds + stage_row * kRowStride +
                    stage_chunk * kPacked) = do_stage;
            }
        }
        __syncthreads();

        Float8 score = {};
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            Half16 q_row;
            if constexpr(kCacheQRows)
            {
                q_row = *reinterpret_cast<const Half16*>(
                    q_row_lds + lane_row * kRowStride +
                    d_tile * kInnerBlock);
            }
            else
            {
                q_row = {};
                if(q_start + lane_row < n_q)
                {
                    const int offset = q_offset +
                                       (q_start + lane_row) * q_row_stride +
                                       d_tile * kInnerBlock;
                    q_row = *reinterpret_cast<const Half16*>(q + offset);
                }
            }
            WmmaInPlace(score, q_row, k_rows[d_tile]);
        }

        const bool lane_q_valid = q_start + lane_row < n_q;
        const float lane_lse = lane_q_valid
                                   ? lse[head_linear * n_q + q_start + lane_row]
                                   : 0.0f;
        const int lane_lse_bits = __builtin_bit_cast(int, lane_lse);
        const bool kv_valid = kv_start + kv_row < n_kv;
        Float8 probability;
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int q_column = i * 2 + lane_group;
            const int even_lse_bits =
                __builtin_amdgcn_readlane(lane_lse_bits, i * 2);
            const int odd_lse_bits =
                __builtin_amdgcn_readlane(lane_lse_bits, i * 2 + 1);
            const float column_lse = __builtin_bit_cast(
                float, lane_group == 0 ? even_lse_bits : odd_lse_bits);
            const bool valid = kv_valid && q_start + q_column < n_q;
            probability[i] =
                valid ? __expf(score[i] * scale - column_lse) : 0.0f;
        }

        if(owns_dk)
        {
            Float8 d_probability = {};
#pragma unroll
            for(int d_tile = 0;
                d_tile < kHeadDim / kInnerBlock;
                ++d_tile)
            {
                Half16 do_row;
                if constexpr(kCacheDoRows)
                {
                    do_row = *reinterpret_cast<const Half16*>(
                        do_row_lds + lane_row * kRowStride +
                        d_tile * kInnerBlock);
                }
                else
                {
                    do_row = {};
                    if(q_start + lane_row < n_q)
                    {
                        const int offset = q_offset +
                                           (q_start + lane_row) * q_row_stride +
                                           d_tile * kInnerBlock;
                        do_row = *reinterpret_cast<const Half16*>(
                            dout + offset);
                    }
                }
                WmmaInPlace(
                    d_probability,
                    do_row,
                    LoadKVRow16(v_lds, kv_row, d_tile));
            }
            const float lane_delta = lane_q_valid
                                         ? delta[head_linear * n_q +
                                                 q_start + lane_row]
                                         : 0.0f;
            const int lane_delta_bits =
                __builtin_bit_cast(int, lane_delta);
            Float8 d_score;
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int q_column = i * 2 + lane_group;
                const int even_delta_bits =
                    __builtin_amdgcn_readlane(lane_delta_bits, i * 2);
                const int odd_delta_bits =
                    __builtin_amdgcn_readlane(lane_delta_bits, i * 2 + 1);
                const float column_delta = __builtin_bit_cast(
                    float,
                    lane_group == 0 ? even_delta_bits : odd_delta_bits);
                const bool valid = kv_valid && q_start + q_column < n_q;
                d_score[i] = valid
                                 ? probability[i] *
                                       (d_probability[i] - column_delta)
                                 : 0.0f;
            }
            const Half16 ds_fragment = CFragmentToA(d_score, lane);
#pragma unroll
            for(int d_pair = 0;
                d_pair < kHeadDim / (2 * kInnerBlock);
                ++d_pair)
            {
                const int d0 = d_pair * 2 * kInnerBlock + lane_row;
                const int d1 = d0 + kInnerBlock;
                WmmaInPlace(
                    grad_accum[d_pair * 2],
                    ds_fragment,
                    LoadTransposeRow16(q_transpose_lds, d0));
                WmmaInPlace(
                    grad_accum[d_pair * 2 + 1],
                    ds_fragment,
                    LoadTransposeRow16(q_transpose_lds, d1));
            }
        }
        else
        {
            const Half16 p_fragment = CFragmentToA(probability, lane);
#pragma unroll
            for(int d_pair = 0;
                d_pair < kHeadDim / (2 * kInnerBlock);
                ++d_pair)
            {
                const int d0 = d_pair * 2 * kInnerBlock + lane_row;
                const int d1 = d0 + kInnerBlock;
                WmmaInPlace(
                    grad_accum[d_pair * 2],
                    p_fragment,
                    LoadTransposeRow16(do_transpose_lds, d0));
                WmmaInPlace(
                    grad_accum[d_pair * 2 + 1],
                    p_fragment,
                    LoadTransposeRow16(do_transpose_lds, d1));
            }
        }
    }

#pragma unroll
    for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
    {
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int row = owner_wave * kInnerBlock +
                            i * 2 + lane_group;
            if(kv_start + row < n_kv)
            {
                const int d = d_tile * kInnerBlock + lane_row;
                const int offset = kv_offset +
                                   (kv_start + row) * kv_row_stride + d;
                if(owns_dk)
                    dk[offset] = __float2half(grad_accum[d_tile][i] * scale);
                else
                    dv[offset] = __float2half(grad_accum[d_tile][i]);
            }
        }
    }
}

template<bool kNhd>
__global__ __launch_bounds__(kQThreads) void D128QKernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ dout,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    __half* __restrict__ dq,
    int n_q,
    int n_kv,
    int num_heads,
    int head_count,
    float scale)
{
    constexpr int tile_bytes =
        kInnerBlock * kHeadDim * sizeof(_Float16);
    constexpr int transpose_bytes =
        kHeadDim * kTransposeStride * sizeof(_Float16);
    constexpr int lds_bytes = 2 * tile_bytes + transpose_bytes;
    static_assert(tile_bytes == 4096);
    static_assert(transpose_bytes == 5120);
    static_assert(lds_bytes == 13312);

    alignas(16) __shared__ uint8_t lds[lds_bytes];
    auto* k_lds = reinterpret_cast<_Float16*>(lds);
    auto* v_lds = reinterpret_cast<_Float16*>(lds + tile_bytes);
    auto* k_transpose_lds = reinterpret_cast<_Float16*>(
        lds + 2 * tile_bytes);

    const int tid = threadIdx.x;
    const int wave = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int lane_row = lane % kInnerBlock;
    const int lane_group = lane / kInnerBlock;
    const int q_tiles = (n_q + kQOwnerBlock - 1) / kQOwnerBlock;
    int owner_tile;
    int head_linear;
    DecodeOwner<kNhd>(
        static_cast<int>(blockIdx.x),
        q_tiles,
        head_count,
        num_heads,
        owner_tile,
        head_linear);
    const int q_block_start = owner_tile * kQOwnerBlock;
    const int q_start = q_block_start + wave * kInnerBlock;
    const int q_offset = HeadOffset<kNhd>(head_linear, n_q, num_heads);
    const int kv_offset = HeadOffset<kNhd>(head_linear, n_kv, num_heads);
    const int q_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;
    const int kv_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;
    const bool q_valid = q_start + lane_row < n_q;
    const float row_lse = q_valid
                              ? lse[head_linear * n_q + q_start + lane_row]
                              : 0.0f;
    const float row_delta = q_valid
                                ? delta[head_linear * n_q + q_start + lane_row]
                                : 0.0f;
    Half16 q_rows[kHeadDim / kInnerBlock] = {};
    Half16 do_rows[kHeadDim / kInnerBlock] = {};
#pragma unroll
    for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
    {
        if(q_valid)
        {
            const int offset = q_offset +
                               (q_start + lane_row) * q_row_stride +
                               d_tile * kInnerBlock;
            q_rows[d_tile] = *reinterpret_cast<const Half16*>(q + offset);
            do_rows[d_tile] = *reinterpret_cast<const Half16*>(dout + offset);
        }
    }
    Float8 dq_accum[kHeadDim / kInnerBlock] = {};

    for(int kv_start = 0; kv_start < n_kv; kv_start += kInnerBlock)
    {
        __syncthreads();
#pragma unroll
        for(int issue = 0; issue < 1; ++issue)
        {
            const int linear_chunk = tid + issue * kQThreads;
            if(linear_chunk < kInnerBlock * kHeadDim / kPacked)
            {
                const int stage_row =
                    linear_chunk / (kHeadDim / kPacked);
                const int stage_chunk =
                    linear_chunk % (kHeadDim / kPacked);
                Half8 k_stage = {};
                Half8 v_stage = {};
                if(kv_start + stage_row < n_kv)
                {
                    const int offset = kv_offset +
                                       (kv_start + stage_row) *
                                           kv_row_stride +
                                       stage_chunk * kPacked;
                    k_stage = *reinterpret_cast<const Half8*>(k + offset);
                    v_stage = *reinterpret_cast<const Half8*>(v + offset);
                }
                *reinterpret_cast<Half8*>(
                    k_lds + KVLdsOffset(stage_row, stage_chunk)) = k_stage;
                *reinterpret_cast<Half8*>(
                    v_lds + KVLdsOffset(stage_row, stage_chunk)) = v_stage;
#pragma unroll
                for(int i = 0; i < kPacked; ++i)
                {
                    const int d = stage_chunk * kPacked + i;
                    k_transpose_lds[d * kTransposeStride + stage_row] =
                        k_stage[i];
                }
            }
        }
        __syncthreads();

        Float8 score = {};
        Float8 d_probability = {};
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            WmmaInPlace(
                score,
                LoadKVRow16(k_lds, lane_row, d_tile),
                q_rows[d_tile]);
            WmmaInPlace(
                d_probability,
                LoadKVRow16(v_lds, lane_row, d_tile),
                do_rows[d_tile]);
        }

        Float8 d_score;
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int kv_column = i * 2 + lane_group;
            const bool valid = q_valid &&
                               kv_start + kv_column < n_kv;
            const float probability =
                valid ? __expf(score[i] * scale - row_lse) : 0.0f;
            d_score[i] = valid
                             ? probability *
                                   (d_probability[i] - row_delta)
                             : 0.0f;
        }
        const Half16 ds_fragment = CFragmentToA(d_score, lane);
#pragma unroll
        for(int d_pair = 0;
            d_pair < kHeadDim / (2 * kInnerBlock);
            ++d_pair)
        {
            const int d0 = d_pair * 2 * kInnerBlock + lane_row;
            const int d1 = d0 + kInnerBlock;
            WmmaInPlace(
                dq_accum[d_pair * 2],
                ds_fragment,
                LoadTransposeRow16(k_transpose_lds, d0));
            WmmaInPlace(
                dq_accum[d_pair * 2 + 1],
                ds_fragment,
                LoadTransposeRow16(k_transpose_lds, d1));
        }
    }

#pragma unroll
    for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
    {
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int row = i * 2 + lane_group;
            if(q_start + row < n_q)
            {
                const int d = d_tile * kInnerBlock + lane_row;
                dq[q_offset + (q_start + row) * q_row_stride + d] =
                    __float2half(dq_accum[d_tile][i] * scale);
            }
        }
    }
}

template<bool kNhd>
bool LaunchD128Backward(const BackwardLaunchParams& params)
{
    const auto* q = reinterpret_cast<const __half*>(params.q_ptr);
    const auto* k = reinterpret_cast<const __half*>(params.k_ptr);
    const auto* v = reinterpret_cast<const __half*>(params.v_ptr);
    const auto* out = reinterpret_cast<const __half*>(params.out_ptr);
    const auto* lse = reinterpret_cast<const float*>(params.lse_ptr);
    const auto* dout = reinterpret_cast<const __half*>(params.dout_ptr);
    auto* dq = reinterpret_cast<__half*>(params.dq_ptr);
    auto* dk = reinterpret_cast<__half*>(params.dk_ptr);
    auto* dv = reinterpret_cast<__half*>(params.dv_ptr);
    auto* delta = reinterpret_cast<float*>(params.delta_ptr);
    const int32_t q_rows = params.head_count * params.n_q;
    const int32_t q_tiles =
        (params.n_q + kQOwnerBlock - 1) / kQOwnerBlock;
    const int32_t kv_tiles =
        (params.n_kv + kKVOwnerBlock - 1) / kKVOwnerBlock;

    (void)hipGetLastError();
    const dim3 delta_block(kRowsBlock);
    const dim3 delta_grid(
        (static_cast<uint32_t>(q_rows) + kRowsBlock - 1) /
        kRowsBlock);
    hipLaunchKernelGGL(
        (DeltaD128Kernel<kNhd>),
        delta_grid,
        delta_block,
        0,
        params.stream,
        out,
        dout,
        delta,
        q_rows,
        params.n_q,
        params.num_heads);

    const dim3 kv_block(kThreads);
    const dim3 kv_grid(
        static_cast<uint32_t>(params.head_count * kv_tiles));
    auto launch_kv = [&]<bool kCacheQRows, bool kCacheDoRows>() {
        hipLaunchKernelGGL(
            (D128KVKernel<kNhd, kCacheQRows, kCacheDoRows>),
            kv_grid,
            kv_block,
            0,
            params.stream,
            q,
            k,
            v,
            dout,
            lse,
            delta,
            dk,
            dv,
            params.n_q,
            params.n_kv,
            params.num_heads,
            params.head_count,
            params.scale);
    };
    if constexpr(kNhd)
    {
        if(params.num_heads <= 16)
            launch_kv.template operator()<false, false>();
        else
            launch_kv.template operator()<false, true>();
    }
    else
    {
        launch_kv.template operator()<false, false>();
    }

    const dim3 q_grid(
        static_cast<uint32_t>(params.head_count * q_tiles));
    const dim3 q_block(kQThreads);
    hipLaunchKernelGGL(
        (D128QKernel<kNhd>),
        q_grid,
        q_block,
        0,
        params.stream,
        q,
        k,
        v,
        dout,
        lse,
        delta,
        dq,
        params.n_q,
        params.n_kv,
        params.num_heads,
        params.head_count,
        params.scale);
    return hipPeekAtLastError() == hipSuccess;
}

} // namespace

extern "C" bool feather_attn_bwd_d128_fused(
    const BackwardLaunchParams& params)
{
    return params.layout == 1 ? LaunchD128Backward<true>(params)
                              : LaunchD128Backward<false>(params);
}

} // namespace feather_attn
