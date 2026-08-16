#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "hip_kernel.h"

#include <cstdint>

namespace feather_attn {
namespace {

constexpr int kRowsBlock = 256;
constexpr int kThreads = 128;
constexpr int kWaveSize = 32;
constexpr int kOwnerBlock = 64;
constexpr int kInnerBlock = 16;
constexpr int kHeadDim = 64;
constexpr int kPacked = 8;
constexpr int kTransposeStride = 20;

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
        words[i * 2] = __builtin_amdgcn_perm(
            peer, value, selector_0);
        words[i * 2 + 1] = __builtin_amdgcn_perm(
            peer, value, selector_1);
    }
    return __builtin_bit_cast(Half16, words);
}

__global__ void DeltaKernel(
    const __half* __restrict__ out,
    const __half* __restrict__ dout,
    float* __restrict__ delta,
    int rows)
{
    const int linear = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear >= rows)
        return;
    const __half* out_row = out + linear * kHeadDim;
    const __half* dout_row = dout + linear * kHeadDim;
    float value = 0.0f;
#pragma unroll
    for(int d = 0; d < kHeadDim; ++d)
        value += __half2float(out_row[d]) * __half2float(dout_row[d]);
    delta[linear] = value;
}

template<int kPhase>
__global__ __launch_bounds__(kThreads) void SevenGemmD64Kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ dout,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    __half* __restrict__ dq,
    __half* __restrict__ dk,
    __half* __restrict__ dv,
    int n_q,
    int n_kv,
    float scale)
{
    constexpr int kv_bytes = kOwnerBlock * kHeadDim * sizeof(_Float16);
    constexpr int transpose_bytes =
        2 * kTransposeStride * kHeadDim * sizeof(_Float16);
    constexpr int q_kv_tile_bytes =
        2 * kInnerBlock * kHeadDim * sizeof(_Float16);
    constexpr int do_row_stride = 72;
    constexpr int do_row_bytes =
        kInnerBlock * do_row_stride * sizeof(_Float16);
    constexpr int lds_bytes =
        kv_bytes + transpose_bytes + (kPhase == 1 ? do_row_bytes : 0);
    static_assert(kv_bytes == 8192);
    static_assert(q_kv_tile_bytes == 4096);
    static_assert(do_row_bytes == 2304);
    static_assert(transpose_bytes == 5120);
    static_assert(lds_bytes == (kPhase == 1 ? 15616 : 13312));

    alignas(16) __shared__ uint8_t lds[lds_bytes];
    auto* kv_v_lds = reinterpret_cast<_Float16*>(lds);
    auto* q_k_lds = reinterpret_cast<_Float16*>(lds);
    auto* q_v_lds = reinterpret_cast<_Float16*>(lds + q_kv_tile_bytes);
    auto* transpose_lds = lds + kv_bytes;
    auto* k_transpose_lds = reinterpret_cast<_Float16*>(transpose_lds);
    auto* q_transpose_lds = reinterpret_cast<_Float16*>(transpose_lds);
    auto* do_transpose_lds = q_transpose_lds + kTransposeStride * kHeadDim;
    auto* do_row_lds =
        reinterpret_cast<_Float16*>(transpose_lds + transpose_bytes);
    static_assert(
        transpose_bytes + (kPhase == 1 ? do_row_bytes : 0) <=
        lds_bytes - kv_bytes);

    const int tid = threadIdx.x;
    const int wave = tid / kWaveSize;
    const int lane = tid % kWaveSize;
    const int lane_row = lane % kInnerBlock;
    const int lane_group = lane / kInnerBlock;
    const int q_tiles = (n_q + kOwnerBlock - 1) / kOwnerBlock;
    const int kv_tiles = (n_kv + kOwnerBlock - 1) / kOwnerBlock;
    const int owner_tiles =
        kPhase == 1 ? kv_tiles
                    : (kPhase == 2 ? q_tiles
                                   : (q_tiles > kv_tiles ? q_tiles : kv_tiles));
    const int owner_tile = static_cast<int>(blockIdx.x) % owner_tiles;
    const int head_linear = static_cast<int>(blockIdx.x) / owner_tiles;
    const int q_offset = head_linear * n_q * kHeadDim;
    const int kv_offset = head_linear * n_kv * kHeadDim;

    if(kPhase != 2 && owner_tile < kv_tiles)
    {
        const int kv_start = owner_tile * kOwnerBlock;
#pragma unroll
        for(int issue = 0; issue < 4; ++issue)
        {
            const int linear_chunk = tid + issue * kThreads;
            const int row = linear_chunk / (kHeadDim / kPacked);
            const int chunk = linear_chunk % (kHeadDim / kPacked);
            Half8 v_value = {};
            if(kv_start + row < n_kv)
            {
                const int global_offset =
                    kv_offset + (kv_start + row) * kHeadDim + chunk * kPacked;
                v_value = *reinterpret_cast<const Half8*>(v + global_offset);
            }
            *reinterpret_cast<Half8*>(kv_v_lds + KVLdsOffset(row, chunk)) =
                v_value;
        }
        __syncthreads();

        Float8 dk_accum[4] = {};
        Float8 dv_accum[4] = {};
        Half16 k_rows[kHeadDim / kInnerBlock];
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            const int kv_row = wave * kInnerBlock + lane_row;
            k_rows[d_tile] = {};
            if(kv_start + kv_row < n_kv)
            {
                const int global_offset =
                    kv_offset + (kv_start + kv_row) * kHeadDim +
                    d_tile * kInnerBlock;
                k_rows[d_tile] = *reinterpret_cast<const Half16*>(
                    k + global_offset);
            }
        }

        for(int q_start = 0; q_start < n_q; q_start += kInnerBlock)
        {
            __syncthreads();
            const int stage_row = tid / (kHeadDim / kPacked);
            const int stage_chunk = tid % (kHeadDim / kPacked);
            Half8 q_stage = {};
            Half8 do_stage = {};
            if(q_start + stage_row < n_q)
            {
                const int global_offset =
                    q_offset + (q_start + stage_row) * kHeadDim +
                    stage_chunk * kPacked;
                q_stage = *reinterpret_cast<const Half8*>(q + global_offset);
                do_stage = *reinterpret_cast<const Half8*>(dout + global_offset);
            }
#pragma unroll
            for(int i = 0; i < kPacked; ++i)
            {
                const int d = stage_chunk * kPacked + i;
                q_transpose_lds[d * kTransposeStride + stage_row] = q_stage[i];
                do_transpose_lds[d * kTransposeStride + stage_row] =
                    do_stage[i];
            }
            if constexpr(kPhase == 1)
            {
                *reinterpret_cast<Half8*>(
                    do_row_lds + stage_row * do_row_stride +
                    stage_chunk * kPacked) = do_stage;
            }
            __syncthreads();

            Float8 score = {};
#pragma unroll
            for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
            {
                Half16 q_row = {};
                if(q_start + lane_row < n_q)
                {
                    const int offset =
                        q_offset + (q_start + lane_row) * kHeadDim +
                        d_tile * 16;
                    q_row = *reinterpret_cast<const Half16*>(q + offset);
                }
                WmmaInPlace(score, q_row, k_rows[d_tile]);
            }

            const bool lane_q_valid = q_start + lane_row < n_q;
            const float lane_lse = lane_q_valid
                                       ? lse[head_linear * n_q + q_start + lane_row]
                                       : 0.0f;
            const int lane_lse_bits = __builtin_bit_cast(int, lane_lse);
            const int kv_row = wave * kInnerBlock + lane_row;
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
            const Half16 p_fragment = CFragmentToA(probability, lane);

#pragma unroll
            for(int d_pair = 0;
                d_pair < kHeadDim / (2 * kInnerBlock);
                ++d_pair)
            {
                const int d0 = d_pair * 2 * kInnerBlock + lane_row;
                const int d1 = d0 + kInnerBlock;
                const Half16 do_column0 =
                    LoadTransposeRow16(do_transpose_lds, d0);
                const Half16 do_column1 =
                    LoadTransposeRow16(do_transpose_lds, d1);
                WmmaInPlace(dv_accum[d_pair * 2], p_fragment, do_column0);
                WmmaInPlace(dv_accum[d_pair * 2 + 1], p_fragment, do_column1);
            }

            Float8 d_probability = {};
#pragma unroll
            for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
            {
                Half16 do_row = {};
                if constexpr(kPhase == 1)
                {
                    do_row = *reinterpret_cast<const Half16*>(
                        do_row_lds + lane_row * do_row_stride +
                        d_tile * kInnerBlock);
                }
                else if(q_start + lane_row < n_q)
                {
                    const int offset =
                        q_offset + (q_start + lane_row) * kHeadDim +
                        d_tile * 16;
                    do_row = *reinterpret_cast<const Half16*>(dout + offset);
                }
                WmmaInPlace(
                    d_probability,
                    do_row,
                    LoadKVRow16(kv_v_lds, kv_row, d_tile));
            }

            const float lane_delta = lane_q_valid
                                         ? delta[head_linear * n_q + q_start + lane_row]
                                         : 0.0f;
            const int lane_delta_bits = __builtin_bit_cast(int, lane_delta);
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
                    float, lane_group == 0 ? even_delta_bits : odd_delta_bits);
                const bool valid = kv_valid && q_start + q_column < n_q;
                const float p = probability[i];
                d_score[i] =
                    valid ? p * (d_probability[i] - column_delta) : 0.0f;
            }
            const Half16 ds_fragment = CFragmentToA(d_score, lane);

#pragma unroll
            for(int d_pair = 0;
                d_pair < kHeadDim / (2 * kInnerBlock);
                ++d_pair)
            {
                const int d0 = d_pair * 2 * kInnerBlock + lane_row;
                const int d1 = d0 + kInnerBlock;
                const Half16 q_column0 =
                    LoadTransposeRow16(q_transpose_lds, d0);
                const Half16 q_column1 =
                    LoadTransposeRow16(q_transpose_lds, d1);
                WmmaInPlace(dk_accum[d_pair * 2], ds_fragment, q_column0);
                WmmaInPlace(dk_accum[d_pair * 2 + 1], ds_fragment, q_column1);
            }
            __builtin_amdgcn_wave_barrier();
        }

#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int row = wave * kInnerBlock + i * 2 + lane_group;
                if(kv_start + row < n_kv)
                {
                    const int d = d_tile * kInnerBlock + lane_row;
                    const int offset =
                        kv_offset + (kv_start + row) * kHeadDim + d;
                    dk[offset] = __float2half(dk_accum[d_tile][i] * scale);
                    dv[offset] = __float2half(dv_accum[d_tile][i]);
                }
            }
        }
    }

    if(kPhase != 1 && owner_tile < q_tiles)
    {
        const int q_block_start = owner_tile * kOwnerBlock;
        const int q_start = q_block_start + wave * kInnerBlock;
        Float8 dq_accum[4] = {};
        Half16 q_rows[kHeadDim / kInnerBlock] = {};
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            if(q_start + lane_row < n_q)
            {
                const int offset =
                    q_offset + (q_start + lane_row) * kHeadDim +
                    d_tile * kInnerBlock;
                q_rows[d_tile] = *reinterpret_cast<const Half16*>(q + offset);
            }
        }
        Half16 do_rows[kHeadDim / kInnerBlock] = {};
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            if(q_start + lane_row < n_q)
            {
                const int offset =
                    q_offset + (q_start + lane_row) * kHeadDim +
                    d_tile * kInnerBlock;
                do_rows[d_tile] = *reinterpret_cast<const Half16*>(
                    dout + offset);
            }
        }
        const bool q_valid = q_start + lane_row < n_q;
        const float row_lse = q_valid
                                  ? lse[head_linear * n_q + q_start + lane_row]
                                  : 0.0f;
        const float row_delta = q_valid
                                    ? delta[head_linear * n_q + q_start + lane_row]
                                    : 0.0f;

        for(int kv_start = 0; kv_start < n_kv;
            kv_start += 2 * kInnerBlock)
        {
            __syncthreads();
#pragma unroll
            for(int issue = 0; issue < 2; ++issue)
            {
                const int linear_chunk = tid + issue * kThreads;
                const int stage_row = linear_chunk / (kHeadDim / kPacked);
                const int stage_chunk = linear_chunk % (kHeadDim / kPacked);
                Half8 k_stage = {};
                Half8 v_stage = {};
                if(kv_start + stage_row < n_kv)
                {
                    const int global_offset =
                        kv_offset + (kv_start + stage_row) * kHeadDim +
                        stage_chunk * kPacked;
                    k_stage = *reinterpret_cast<const Half8*>(k + global_offset);
                    v_stage = *reinterpret_cast<const Half8*>(v + global_offset);
                }
                *reinterpret_cast<Half8*>(
                    q_k_lds + KVLdsOffset(stage_row, stage_chunk)) = k_stage;
                *reinterpret_cast<Half8*>(
                    q_v_lds + KVLdsOffset(stage_row, stage_chunk)) = v_stage;
#pragma unroll
                for(int i = 0; i < kPacked; ++i)
                {
                    const int d = stage_chunk * kPacked + i;
                    const int tile = stage_row / kInnerBlock;
                    const int tile_row = stage_row % kInnerBlock;
                    k_transpose_lds[
                        tile * kHeadDim * kTransposeStride +
                        d * kTransposeStride + tile_row] = k_stage[i];
                }
            }
            __syncthreads();

#pragma unroll
            for(int tile = 0; tile < 2; ++tile)
            {
                const int tile_kv_start = kv_start + tile * kInnerBlock;
                const int lds_row = tile * kInnerBlock + lane_row;
                Float8 score = {};
#pragma unroll
                for(int d_tile = 0;
                    d_tile < kHeadDim / kInnerBlock;
                    ++d_tile)
                {
                    const Half16 k_row =
                        LoadKVRow16(q_k_lds, lds_row, d_tile);
                    WmmaInPlace(score, k_row, q_rows[d_tile]);
                }

                Float8 probability;
#pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const int kv_column = i * 2 + lane_group;
                    const bool valid =
                        q_valid && tile_kv_start + kv_column < n_kv;
                    probability[i] =
                        valid ? __expf(score[i] * scale - row_lse) : 0.0f;
                }
                Float8 d_probability = {};
#pragma unroll
                for(int d_tile = 0;
                    d_tile < kHeadDim / kInnerBlock;
                    ++d_tile)
                {
                    const Half16 v_row =
                        LoadKVRow16(q_v_lds, lds_row, d_tile);
                    WmmaInPlace(d_probability, v_row, do_rows[d_tile]);
                }

                Float8 d_score;
#pragma unroll
                for(int i = 0; i < 8; ++i)
                {
                    const int kv_column = i * 2 + lane_group;
                    const bool valid =
                        q_valid && tile_kv_start + kv_column < n_kv;
                    const float p = probability[i];
                    d_score[i] = valid
                                     ? p * (d_probability[i] - row_delta)
                                     : 0.0f;
                }
                const Half16 ds_fragment = CFragmentToA(d_score, lane);

#pragma unroll
                for(int d_tile = 0;
                    d_tile < kHeadDim / kInnerBlock;
                    ++d_tile)
                {
                    const int d = d_tile * kInnerBlock + lane_row;
                    const Half16 k_column = LoadTransposeRow16(
                        k_transpose_lds + tile * kHeadDim * kTransposeStride,
                        d);
                    WmmaInPlace(dq_accum[d_tile], ds_fragment, k_column);
                }
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
                    dq[q_offset + (q_start + row) * kHeadDim + d] =
                        __float2half(dq_accum[d_tile][i] * scale);
                }
            }
        }
    }
}

bool LaunchSevenGemmBackward(const BackwardLaunchParams& params)
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
        (params.n_q + kOwnerBlock - 1) / kOwnerBlock;
    const int32_t kv_tiles =
        (params.n_kv + kOwnerBlock - 1) / kOwnerBlock;
    const int32_t owner_tiles = q_tiles > kv_tiles ? q_tiles : kv_tiles;

    (void)hipGetLastError();
    const dim3 delta_block(kRowsBlock);
    const dim3 delta_grid(
        (static_cast<uint32_t>(q_rows) + kRowsBlock - 1) /
        kRowsBlock);
    hipLaunchKernelGGL(
        DeltaKernel,
        delta_grid,
        delta_block,
        0,
        params.stream,
        out,
        dout,
        delta,
        q_rows);

    const dim3 main_block(kThreads);
    if(params.n_q >= 4096 && params.n_kv >= 4096)
    {
        const dim3 kv_grid(
            static_cast<uint32_t>(params.head_count * kv_tiles));
        hipLaunchKernelGGL(
            (SevenGemmD64Kernel<1>),
            kv_grid,
            main_block,
            0,
            params.stream,
            q,
            k,
            v,
            dout,
            lse,
            delta,
            dq,
            dk,
            dv,
            params.n_q,
            params.n_kv,
            params.scale);

        const dim3 q_grid(
            static_cast<uint32_t>(params.head_count * q_tiles));
        hipLaunchKernelGGL(
            (SevenGemmD64Kernel<2>),
            q_grid,
            main_block,
            0,
            params.stream,
            q,
            k,
            v,
            dout,
            lse,
            delta,
            dq,
            dk,
            dv,
            params.n_q,
            params.n_kv,
            params.scale);
    }
    else
    {
        const dim3 main_grid(
            static_cast<uint32_t>(params.head_count * owner_tiles));
        hipLaunchKernelGGL(
            (SevenGemmD64Kernel<0>),
            main_grid,
            main_block,
            0,
            params.stream,
            q,
            k,
            v,
            dout,
            lse,
            delta,
            dq,
            dk,
            dv,
            params.n_q,
            params.n_kv,
            params.scale);
    }
    return hipPeekAtLastError() == hipSuccess;
}

} // namespace

extern "C" bool feather_attn_bwd_d64_fused(
    const BackwardLaunchParams& params)
{
    return LaunchSevenGemmBackward(params);
}

} // namespace feather_attn
