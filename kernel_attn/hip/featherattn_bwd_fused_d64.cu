#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "hip_kernel.h"

#include <cstdint>

namespace feather_attn {
namespace {

constexpr int kRowsBlock = 256;
constexpr int kThreads = 128;
constexpr int kKVThreads = 256;
constexpr int kQThreads = 256;
constexpr int kWaveSize = 32;
constexpr int kOwnerBlock = 64;
constexpr int kKVOwnerBlock = 128;
constexpr int kQOwnerBlock = 128;
constexpr int kInnerBlock = 16;
constexpr int kHeadDim = 64;
constexpr int kPacked = 8;
constexpr int kTransposeStride = 20;
constexpr float kLog2E = 0x1.715476p+0f;

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
__global__ void DeltaKernel(
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
    int physical_row = linear;
    int delta_row = linear;
    if constexpr(kNhd)
    {
        const int batch = linear / (n_q * num_heads);
        const int batch_row = linear - batch * n_q * num_heads;
        const int q_row = batch_row / num_heads;
        const int head = batch_row - q_row * num_heads;
        delta_row = (batch * num_heads + head) * n_q + q_row;
    }
    const int offset = physical_row * kHeadDim;
    const __half* out_row = out + offset;
    const __half* dout_row = dout + offset;
    float value = 0.0f;
#pragma unroll
    for(int d = 0; d < kHeadDim; ++d)
        value += __half2float(out_row[d]) * __half2float(dout_row[d]);
    delta[delta_row] = value;
}

template<int kPhase, bool kNhd>
__global__ __launch_bounds__(
    kPhase == 1 && !kNhd
        ? kKVThreads
        : (kPhase == 2 ? kQThreads : kThreads)) void
SevenGemmD64Kernel(
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
    int num_heads,
    int head_count,
    int nhd_group_count,
    float scale)
{
    constexpr bool wide_kv = kPhase == 1 && !kNhd;
    constexpr int phase_threads =
        wide_kv ? kKVThreads : (kPhase == 2 ? kQThreads : kThreads);
    constexpr int kv_owner_block =
        wide_kv ? kKVOwnerBlock : kOwnerBlock;
    constexpr int q_owner_block =
        kPhase == 2 ? kQOwnerBlock : kOwnerBlock;
    constexpr int kv_bytes =
        kv_owner_block * kHeadDim * sizeof(_Float16);
    constexpr int transpose_bytes =
        2 * kTransposeStride * kHeadDim * sizeof(_Float16);
    constexpr int q_kv_tile_bytes =
        2 * kInnerBlock * kHeadDim * sizeof(_Float16);
    constexpr int do_row_stride = 72;
    constexpr int do_row_bytes =
        kInnerBlock * do_row_stride * sizeof(_Float16);
    constexpr int lds_bytes =
        kv_bytes + transpose_bytes + (kPhase == 1 ? do_row_bytes : 0);
    static_assert(kv_bytes == (wide_kv ? 16384 : 8192));
    static_assert(q_kv_tile_bytes == 4096);
    static_assert(do_row_bytes == 2304);
    static_assert(transpose_bytes == 5120);
    static_assert(
        lds_bytes == (kPhase == 1 ? (wide_kv ? 23808 : 15616) : 13312));

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
    const int q_tiles = (n_q + q_owner_block - 1) / q_owner_block;
    const int kv_tiles = (n_kv + kv_owner_block - 1) / kv_owner_block;
    const int owner_tiles =
        kPhase == 1 ? kv_tiles
                    : (kPhase == 2 ? q_tiles
                                   : (q_tiles > kv_tiles ? q_tiles : kv_tiles));
    int owner_tile;
    int head_linear;
    if constexpr(kNhd)
    {
        if(nhd_group_count > 1 && num_heads % 16 == 0)
        {
            const int head_slot = static_cast<int>(blockIdx.x) % head_count;
            const int batch = head_slot / num_heads;
            const int slot = head_slot - batch * num_heads;
            const int small_group_size = num_heads / nhd_group_count;
            const int large_groups = num_heads % nhd_group_count;
            const int large_group_size = small_group_size + 1;
            const int large_span = large_groups * large_group_size;
            int group;
            int local_head;
            if(slot < large_span)
            {
                group = slot / large_group_size;
                local_head = slot - group * large_group_size;
            }
            else
            {
                const int small_slot = slot - large_span;
                group = large_groups + small_slot / small_group_size;
                local_head = small_slot % small_group_size;
            }
            head_linear = batch * num_heads + group +
                          local_head * nhd_group_count;
            owner_tile = static_cast<int>(blockIdx.x) / head_count;
        }
        else if(num_heads % 16 == 0)
        {
            head_linear = static_cast<int>(blockIdx.x) % head_count;
            owner_tile = static_cast<int>(blockIdx.x) / head_count;
        }
        else
        {
            owner_tile = static_cast<int>(blockIdx.x) % owner_tiles;
            head_linear = static_cast<int>(blockIdx.x) / owner_tiles;
        }
    }
    else
    {
        owner_tile = static_cast<int>(blockIdx.x) % owner_tiles;
        head_linear = static_cast<int>(blockIdx.x) / owner_tiles;
    }
    const int q_offset = HeadOffset<kNhd>(head_linear, n_q, num_heads);
    const int kv_offset = HeadOffset<kNhd>(head_linear, n_kv, num_heads);
    const int q_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;
    const int kv_row_stride = kNhd ? num_heads * kHeadDim : kHeadDim;

    if(kPhase != 2 && owner_tile < kv_tiles)
    {
        const int kv_start = owner_tile * kv_owner_block;
        const float scale_log2 = scale * kLog2E;
#pragma unroll
        for(int issue = 0;
            issue < kv_owner_block * kHeadDim / kPacked / phase_threads;
            ++issue)
        {
            const int linear_chunk = tid + issue * phase_threads;
            const int row = linear_chunk / (kHeadDim / kPacked);
            const int chunk = linear_chunk % (kHeadDim / kPacked);
            Half8 v_value = {};
            if(kv_start + row < n_kv)
            {
                const int global_offset =
                    kv_offset + (kv_start + row) * kv_row_stride +
                    chunk * kPacked;
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
                    kv_offset + (kv_start + kv_row) * kv_row_stride +
                    d_tile * kInnerBlock;
                k_rows[d_tile] = *reinterpret_cast<const Half16*>(
                    k + global_offset);
            }
        }

        for(int q_start = 0; q_start < n_q; q_start += kInnerBlock)
        {
            __syncthreads();
            if(tid < kInnerBlock * kHeadDim / kPacked)
            {
                const int stage_row = tid / (kHeadDim / kPacked);
                const int stage_chunk = tid % (kHeadDim / kPacked);
                Half8 q_stage = {};
                Half8 do_stage = {};
                if(q_start + stage_row < n_q)
                {
                    const int global_offset =
                        q_offset + (q_start + stage_row) * q_row_stride +
                        stage_chunk * kPacked;
                    q_stage = *reinterpret_cast<const Half8*>(q + global_offset);
                    do_stage = *reinterpret_cast<const Half8*>(
                        dout + global_offset);
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
                if constexpr(kPhase == 1)
                {
                    *reinterpret_cast<Half8*>(
                        do_row_lds + stage_row * do_row_stride +
                        stage_chunk * kPacked) = do_stage;
                }
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
                        q_offset + (q_start + lane_row) * q_row_stride +
                        d_tile * 16;
                    q_row = *reinterpret_cast<const Half16*>(q + offset);
                }
                WmmaInPlace(score, q_row, k_rows[d_tile]);
            }

            const bool lane_q_valid = q_start + lane_row < n_q;
            const float lane_lse = lane_q_valid
                                       ? lse[head_linear * n_q + q_start + lane_row]
                                       : 0.0f;
            const int lane_lse_bits =
                __builtin_bit_cast(int, lane_lse * kLog2E);
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
                const float column_lse_log2 = __builtin_bit_cast(
                    float, lane_group == 0 ? even_lse_bits : odd_lse_bits);
                const bool valid = kv_valid && q_start + q_column < n_q;
                probability[i] =
                    valid
                        ? __builtin_amdgcn_exp2f(
                              score[i] * scale_log2 - column_lse_log2)
                        : 0.0f;
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
                        q_offset + (q_start + lane_row) * q_row_stride +
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
                        kv_offset + (kv_start + row) * kv_row_stride + d;
                    dk[offset] = __float2half(dk_accum[d_tile][i] * scale);
                    dv[offset] = __float2half(dv_accum[d_tile][i]);
                }
            }
        }
    }

    if(kPhase != 1 && owner_tile < q_tiles)
    {
        const int q_block_start = owner_tile * q_owner_block;
        const int q_start = q_block_start + wave * kInnerBlock;
        const float scale_log2 = scale * kLog2E;
        Float8 dq_accum[4] = {};
        Half16 q_rows[kHeadDim / kInnerBlock] = {};
#pragma unroll
        for(int d_tile = 0; d_tile < kHeadDim / kInnerBlock; ++d_tile)
        {
            if(q_start + lane_row < n_q)
            {
                const int offset =
                    q_offset + (q_start + lane_row) * q_row_stride +
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
                    q_offset + (q_start + lane_row) * q_row_stride +
                    d_tile * kInnerBlock;
                do_rows[d_tile] = *reinterpret_cast<const Half16*>(
                    dout + offset);
            }
        }
        const bool q_valid = q_start + lane_row < n_q;
        const float row_lse_log2 =
            q_valid
                ? lse[head_linear * n_q + q_start + lane_row] * kLog2E
                : 0.0f;
        const float row_delta = q_valid
                                    ? delta[head_linear * n_q + q_start + lane_row]
                                    : 0.0f;

        for(int kv_start = 0; kv_start < n_kv;
            kv_start += 2 * kInnerBlock)
        {
            __syncthreads();
#pragma unroll
            for(int issue = 0;
                issue < 2 * kInnerBlock * kHeadDim / kPacked / phase_threads;
                ++issue)
            {
                const int linear_chunk = tid + issue * phase_threads;
                const int stage_row = linear_chunk / (kHeadDim / kPacked);
                const int stage_chunk = linear_chunk % (kHeadDim / kPacked);
                Half8 k_stage = {};
                Half8 v_stage = {};
                if(kv_start + stage_row < n_kv)
                {
                    const int global_offset =
                        kv_offset + (kv_start + stage_row) * kv_row_stride +
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
                        valid
                            ? __builtin_amdgcn_exp2f(
                                  score[i] * scale_log2 - row_lse_log2)
                            : 0.0f;
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
                    dq[q_offset + (q_start + row) * q_row_stride + d] =
                        __float2half(dq_accum[d_tile][i] * scale);
                }
            }
        }
    }
}

template<bool kNhd>
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
    const int32_t q_phase_tiles =
        (params.n_q + kQOwnerBlock - 1) / kQOwnerBlock;
    const int32_t kv_tiles =
        (params.n_kv + kOwnerBlock - 1) / kOwnerBlock;
    const int32_t kv_phase_tiles =
        (params.n_kv + kKVOwnerBlock - 1) / kKVOwnerBlock;
    const int32_t owner_tiles = q_tiles > kv_tiles ? q_tiles : kv_tiles;

    (void)hipGetLastError();
    const dim3 delta_block(kRowsBlock);
    const dim3 delta_grid(
        (static_cast<uint32_t>(q_rows) + kRowsBlock - 1) /
        kRowsBlock);
    hipLaunchKernelGGL(
        (DeltaKernel<kNhd>),
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

    const dim3 main_block(kThreads);
    if(params.n_q >= 4096 && params.n_kv >= 4096)
    {
        const dim3 kv_grid(
            static_cast<uint32_t>(
                params.head_count * (kNhd ? kv_tiles : kv_phase_tiles)));
        const dim3 kv_block(kNhd ? kThreads : kKVThreads);
        hipLaunchKernelGGL(
            (SevenGemmD64Kernel<1, kNhd>),
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
            dq,
            dk,
            dv,
            params.n_q,
            params.n_kv,
            params.num_heads,
            params.head_count,
            params.nhd_group_count,
            params.scale);

        const dim3 q_grid(
            static_cast<uint32_t>(params.head_count * q_phase_tiles));
        const dim3 q_block(kQThreads);
        hipLaunchKernelGGL(
            (SevenGemmD64Kernel<2, kNhd>),
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
            dk,
            dv,
            params.n_q,
            params.n_kv,
            params.num_heads,
            params.head_count,
            params.nhd_group_count,
            params.scale);
    }
    else
    {
        const dim3 main_grid(
            static_cast<uint32_t>(params.head_count * owner_tiles));
        hipLaunchKernelGGL(
            (SevenGemmD64Kernel<0, kNhd>),
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
            params.num_heads,
            params.head_count,
            params.nhd_group_count,
            params.scale);
    }
    return hipPeekAtLastError() == hipSuccess;
}

} // namespace

extern "C" bool feather_attn_bwd_d64_fused(
    const BackwardLaunchParams& params)
{
    return params.layout == 1 ? LaunchSevenGemmBackward<true>(params)
                              : LaunchSevenGemmBackward<false>(params);
}

} // namespace feather_attn
