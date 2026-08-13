#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "hip_kernel.h"

#include <cstdint>

namespace feather_attn {

constexpr int kBackwardBlockSize = 256;
constexpr int kFusedBlockSize = 64;
constexpr int kFusedBlockQ = 16;
constexpr int kFusedBlockKV = 32;
constexpr int kFusedHeadDim = 64;
constexpr int kKVPacked = 8;
constexpr int kQRowStride = 72;
constexpr int kPDsRowStride = 17;

using Half8 = _Float16 __attribute__((ext_vector_type(8)));
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));

template <typename Kernel, typename... Args>
void LaunchBackwardRows(
    Kernel kernel,
    int32_t rows,
    hipStream_t stream,
    Args... args)
{
    const dim3 block(kBackwardBlockSize);
    const dim3 grid(
        (static_cast<uint32_t>(rows) + kBackwardBlockSize - 1) /
        kBackwardBlockSize);
    hipLaunchKernelGGL(kernel, grid, block, 0, stream, args...);
}

__device__ inline float LoadHalf(const __half* ptr, int32_t index)
{
    return __half2float(ptr[index]);
}

template <int D>
__device__ inline float DotHalf(
    const __half* lhs,
    const __half* rhs)
{
    float result = 0.0f;
#pragma unroll
    for(int d = 0; d < D; ++d)
        result += LoadHalf(lhs, d) * LoadHalf(rhs, d);
    return result;
}

template <int D>
__global__ void ReferenceDeltaKernel(
    const __half* __restrict__ out,
    const __half* __restrict__ dout,
    float* __restrict__ delta,
    int32_t rows)
{
    const int32_t linear =
        static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear >= rows)
        return;

    const __half* out_row = out + linear * D;
    const __half* dout_row = dout + linear * D;

    float value = 0.0f;
#pragma unroll
    for(int d = 0; d < D; ++d)
        value += LoadHalf(out_row, d) * LoadHalf(dout_row, d);
    delta[linear] = value;
}

template <int D>
__global__ void ReferenceDqKernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ dout,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    __half* __restrict__ dq,
    int32_t rows,
    int32_t n_q,
    int32_t n_kv,
    float scale)
{
    const int32_t linear =
        static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear >= rows)
        return;

    const int32_t q_row = linear % n_q;
    const int32_t head_linear = linear / n_q;
    const __half* q_row_ptr = q + (head_linear * n_q + q_row) * D;
    const __half* dout_row_ptr = dout + (head_linear * n_q + q_row) * D;
    __half* dq_row_ptr = dq + (head_linear * n_q + q_row) * D;
    const float row_lse = lse[linear];
    const float row_delta = delta[linear];

    float accum[D] = {};
    for(int32_t kv_row = 0; kv_row < n_kv; ++kv_row)
    {
        const __half* k_row_ptr = k + (head_linear * n_kv + kv_row) * D;
        const __half* v_row_ptr = v + (head_linear * n_kv + kv_row) * D;
        const float score = DotHalf<D>(q_row_ptr, k_row_ptr) * scale;
        const float probability = __expf(score - row_lse);
        const float d_probability = DotHalf<D>(dout_row_ptr, v_row_ptr);
        const float d_score = probability * (d_probability - row_delta);
#pragma unroll
        for(int d = 0; d < D; ++d)
            accum[d] += d_score * LoadHalf(k_row_ptr, d) * scale;
    }

#pragma unroll
    for(int d = 0; d < D; ++d)
        dq_row_ptr[d] = __float2half(accum[d]);
}

template <int D>
__global__ void ReferenceDkKernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ dout,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    __half* __restrict__ dk,
    int32_t rows,
    int32_t n_q,
    int32_t n_kv,
    float scale)
{
    const int32_t linear =
        static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear >= rows)
        return;

    const int32_t kv_row = linear % n_kv;
    const int32_t head_linear = linear / n_kv;
    const __half* k_row_ptr = k + (head_linear * n_kv + kv_row) * D;
    const __half* v_row_ptr = v + (head_linear * n_kv + kv_row) * D;
    __half* dk_row_ptr = dk + (head_linear * n_kv + kv_row) * D;

    float accum[D] = {};
    for(int32_t q_row = 0; q_row < n_q; ++q_row)
    {
        const __half* q_row_ptr = q + (head_linear * n_q + q_row) * D;
        const __half* dout_row_ptr = dout + (head_linear * n_q + q_row) * D;
        const float score = DotHalf<D>(q_row_ptr, k_row_ptr) * scale;
        const float probability = __expf(score - lse[head_linear * n_q + q_row]);
        const float d_probability = DotHalf<D>(dout_row_ptr, v_row_ptr);
        const float d_score = probability *
                              (d_probability - delta[head_linear * n_q + q_row]);
#pragma unroll
        for(int d = 0; d < D; ++d)
            accum[d] += d_score * LoadHalf(q_row_ptr, d) * scale;
    }

#pragma unroll
    for(int d = 0; d < D; ++d)
        dk_row_ptr[d] = __float2half(accum[d]);
}

template <int D>
__global__ void ReferenceDvKernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ dout,
    const float* __restrict__ lse,
    __half* __restrict__ dv,
    int32_t rows,
    int32_t n_q,
    int32_t n_kv,
    float scale)
{
    const int32_t linear =
        static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear >= rows)
        return;

    const int32_t kv_row = linear % n_kv;
    const int32_t head_linear = linear / n_kv;
    const __half* k_row_ptr = k + (head_linear * n_kv + kv_row) * D;
    __half* dv_row_ptr = dv + (head_linear * n_kv + kv_row) * D;

    float accum[D] = {};
    for(int32_t q_row = 0; q_row < n_q; ++q_row)
    {
        const __half* q_row_ptr = q + (head_linear * n_q + q_row) * D;
        const __half* dout_row_ptr = dout + (head_linear * n_q + q_row) * D;
        const float score = DotHalf<D>(q_row_ptr, k_row_ptr) * scale;
        const float probability = __expf(score - lse[head_linear * n_q + q_row]);
#pragma unroll
        for(int d = 0; d < D; ++d)
            accum[d] += probability * LoadHalf(dout_row_ptr, d);
    }

#pragma unroll
    for(int d = 0; d < D; ++d)
        dv_row_ptr[d] = __float2half(accum[d]);
}

template <int D>
bool LaunchReferenceBackward(const BackwardLaunchParams& params)
{
    static_assert(D == 128);
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
    const int32_t kv_rows = params.head_count * params.n_kv;

    (void)hipGetLastError();
    LaunchBackwardRows(
        ReferenceDeltaKernel<D>, q_rows, params.stream,
        out, dout, delta, q_rows);
    LaunchBackwardRows(
        ReferenceDqKernel<D>, q_rows, params.stream,
        q, k, v, dout, lse, delta, dq,
        q_rows, params.n_q, params.n_kv, params.scale);
    LaunchBackwardRows(
        ReferenceDkKernel<D>, kv_rows, params.stream,
        q, k, v, dout, lse, delta, dk,
        kv_rows, params.n_q, params.n_kv, params.scale);
    LaunchBackwardRows(
        ReferenceDvKernel<D>, kv_rows, params.stream,
        q, k, dout, lse, dv,
        kv_rows, params.n_q, params.n_kv, params.scale);
    return hipPeekAtLastError() == hipSuccess;
}

__device__ inline int32_t KVLdsOffset(int32_t row, int32_t chunk)
{
    constexpr int32_t chunks = kFusedHeadDim / kKVPacked;
    const int32_t physical_chunk = chunk ^ (row % chunks);
    return row * kFusedHeadDim + physical_chunk * kKVPacked;
}

__device__ inline int32_t KVLdsElementOffset(int32_t row, int32_t column)
{
    return KVLdsOffset(row, column / kKVPacked) + column % kKVPacked;
}

__device__ inline Half16 LoadKVRow16(
    const _Float16* ptr,
    int32_t row,
    int32_t d_tile)
{
    const int32_t chunk = d_tile * 2;
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

template <int D>
__global__ void FusedDeltaClearKernel(
    const __half* __restrict__ out,
    const __half* __restrict__ dout,
    float* __restrict__ delta,
    float* __restrict__ dq_acc,
    int32_t rows)
{
    static_assert(D == kFusedHeadDim);
    const int32_t linear =
        static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear >= rows)
        return;

    const __half* out_row = out + linear * D;
    const __half* dout_row = dout + linear * D;
    float* dq_acc_row = dq_acc + linear * D;
    float value = 0.0f;
#pragma unroll
    for(int d = 0; d < D; ++d)
    {
        value += __half2float(out_row[d]) * __half2float(dout_row[d]);
        dq_acc_row[d] = 0.0f;
    }
    delta[linear] = value;
}

template <int D>
__global__ __launch_bounds__(kFusedBlockSize) void FusedBackwardKernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ dout,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    float* __restrict__ dq_acc,
    float* __restrict__ dk_acc,
    float* __restrict__ dv_acc,
    int32_t n_q,
    int32_t n_kv,
    float scale)
{
    static_assert(D == kFusedHeadDim);
    constexpr int32_t kv_chunks = D / kKVPacked;
    constexpr int32_t kv_elements = kFusedBlockKV * D;
    constexpr int32_t q_elements = kFusedBlockQ * kQRowStride;
    constexpr int32_t p_elements = kFusedBlockKV * kPDsRowStride;
    constexpr int32_t k_bytes = kv_elements * sizeof(_Float16);
    constexpr int32_t v_bytes = kv_elements * sizeof(_Float16);
    constexpr int32_t q_bytes = q_elements * sizeof(_Float16);
    constexpr int32_t do_bytes = q_elements * sizeof(_Float16);
    constexpr int32_t p_bytes = p_elements * sizeof(float);
    constexpr int32_t ds_bytes = p_elements * sizeof(float);
    constexpr int32_t lds_bytes =
        k_bytes + v_bytes + q_bytes + do_bytes + p_bytes + ds_bytes;
    static_assert(lds_bytes == 17152);

    alignas(16) __shared__ uint8_t lds[lds_bytes];
    auto* k_lds = reinterpret_cast<_Float16*>(lds);
    auto* v_lds = reinterpret_cast<_Float16*>(lds + k_bytes);
    auto* q_lds = reinterpret_cast<_Float16*>(lds + k_bytes + v_bytes);
    auto* do_lds = reinterpret_cast<_Float16*>(
        lds + k_bytes + v_bytes + q_bytes);
    auto* p_lds = reinterpret_cast<float*>(
        lds + k_bytes + v_bytes + q_bytes + do_bytes);
    auto* ds_lds = reinterpret_cast<float*>(
        lds + k_bytes + v_bytes + q_bytes + do_bytes + p_bytes);

    const int32_t tid = threadIdx.x;
    const int32_t wave = tid / 32;
    const int32_t lane = tid % 32;
    const int32_t lane_row = lane % 16;
    const int32_t lane_group = lane / 16;
    const int32_t kv_tiles = (n_kv + kFusedBlockKV - 1) / kFusedBlockKV;
    const int32_t kv_tile = static_cast<int32_t>(blockIdx.x) % kv_tiles;
    const int32_t head_linear = static_cast<int32_t>(blockIdx.x) / kv_tiles;
    const int32_t kv_start = kv_tile * kFusedBlockKV;
    const int32_t q_head_offset = head_linear * n_q * D;
    const int32_t kv_head_offset = head_linear * n_kv * D;

#pragma unroll
    for(int issue = 0; issue < 4; ++issue)
    {
        const int32_t linear_chunk = tid + issue * kFusedBlockSize;
        const int32_t row = linear_chunk / kv_chunks;
        const int32_t chunk = linear_chunk % kv_chunks;
        Half8 k_value = {};
        Half8 v_value = {};
        if(kv_start + row < n_kv)
        {
            const int32_t global_offset =
                kv_head_offset + (kv_start + row) * D + chunk * kKVPacked;
            k_value = *reinterpret_cast<const Half8*>(k + global_offset);
            v_value = *reinterpret_cast<const Half8*>(v + global_offset);
        }
        const int32_t lds_offset = KVLdsOffset(row, chunk);
        *reinterpret_cast<Half8*>(k_lds + lds_offset) = k_value;
        *reinterpret_cast<Half8*>(v_lds + lds_offset) = v_value;
    }
    __syncthreads();

    for(int32_t q_start = 0; q_start < n_q; q_start += kFusedBlockQ)
    {
#pragma unroll
        for(int issue = 0; issue < 2; ++issue)
        {
            const int32_t q_linear = tid + issue * kFusedBlockSize;
            const int32_t q_row = q_linear / kv_chunks;
            const int32_t d_chunk = q_linear % kv_chunks;
            Half8 q_value = {};
            Half8 do_value = {};
            if(q_start + q_row < n_q)
            {
                const int32_t global_offset =
                    q_head_offset + (q_start + q_row) * D +
                    d_chunk * kKVPacked;
                q_value = *reinterpret_cast<const Half8*>(q + global_offset);
                do_value = *reinterpret_cast<const Half8*>(dout + global_offset);
            }
            *reinterpret_cast<Half8*>(
                q_lds + q_row * kQRowStride + d_chunk * kKVPacked) = q_value;
            *reinterpret_cast<Half8*>(
                do_lds + q_row * kQRowStride + d_chunk * kKVPacked) = do_value;
        }
        __syncthreads();

        Float8 score = {};
        Float8 d_probability = {};
#pragma unroll
        for(int d_tile = 0; d_tile < D / 16; ++d_tile)
        {
            const Half16 q_fragment = *reinterpret_cast<const Half16*>(
                q_lds + lane_row * kQRowStride + d_tile * 16);
            const Half16 do_fragment = *reinterpret_cast<const Half16*>(
                do_lds + lane_row * kQRowStride + d_tile * 16);
            const int32_t kv_row = wave * 16 + lane_row;
            const Half16 k_fragment = LoadKVRow16(k_lds, kv_row, d_tile);
            const Half16 v_fragment = LoadKVRow16(v_lds, kv_row, d_tile);
            WmmaInPlace(score, k_fragment, q_fragment);
            WmmaInPlace(d_probability, v_fragment, do_fragment);
        }

        const bool q_valid = q_start + lane_row < n_q;
        const float row_lse =
            q_valid ? lse[head_linear * n_q + q_start + lane_row] : 0.0f;
        const float row_delta =
            q_valid ? delta[head_linear * n_q + q_start + lane_row] : 0.0f;
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            const int32_t kv_row = wave * 16 + i * 2 + lane_group;
            const bool valid = q_valid && kv_start + kv_row < n_kv;
            float probability = 0.0f;
            float d_score = 0.0f;
            if(valid)
            {
                probability = __expf(score[i] * scale - row_lse);
                d_score = probability * (d_probability[i] - row_delta);
            }
            p_lds[kv_row * kPDsRowStride + lane_row] = probability;
            ds_lds[kv_row * kPDsRowStride + lane_row] = d_score;
        }
        __syncthreads();

#pragma unroll
        for(int d_tile = 0; d_tile < D / 16; ++d_tile)
        {
            Float8 dk_fragment = {};
            Float8 dv_fragment = {};
            Half16 p_fragment;
            Half16 ds_fragment;
            Half16 q_fragment;
            Half16 do_fragment;
            const int32_t kv_row = wave * 16 + lane_row;
            const int32_t d_row = d_tile * 16 + lane_row;
#pragma unroll
            for(int j = 0; j < 16; ++j)
            {
                p_fragment[j] = static_cast<_Float16>(
                    p_lds[kv_row * kPDsRowStride + j]);
                ds_fragment[j] = static_cast<_Float16>(
                    ds_lds[kv_row * kPDsRowStride + j]);
                q_fragment[j] = q_lds[j * kQRowStride + d_row];
                do_fragment[j] = do_lds[j * kQRowStride + d_row];
            }
            WmmaInPlace(dk_fragment, ds_fragment, q_fragment);
            WmmaInPlace(dv_fragment, p_fragment, do_fragment);
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int32_t row = wave * 16 + i * 2 + lane_group;
                if(kv_start + row < n_kv)
                {
                    const int32_t d = d_tile * 16 + lane_row;
                    const int32_t output_offset =
                        kv_head_offset + (kv_start + row) * D + d;
                    const float dk_value = dk_fragment[i] * scale;
                    const float dv_value = dv_fragment[i];
                    if(q_start == 0)
                    {
                        dk_acc[output_offset] = dk_value;
                        dv_acc[output_offset] = dv_value;
                    }
                    else
                    {
                        dk_acc[output_offset] += dk_value;
                        dv_acc[output_offset] += dv_value;
                    }
                }
            }
        }

#pragma unroll
        for(int d_subtile = 0; d_subtile < 2; ++d_subtile)
        {
            Float8 dq_partial = {};
#pragma unroll
            for(int n_tile = 0; n_tile < kFusedBlockKV / 16; ++n_tile)
            {
                Half16 ds_fragment;
                Half16 k_fragment;
                const int32_t d = (wave * 2 + d_subtile) * 16 + lane_row;
#pragma unroll
                for(int j = 0; j < 16; ++j)
                {
                    const int32_t kv_row = n_tile * 16 + j;
                    ds_fragment[j] = static_cast<_Float16>(
                        ds_lds[kv_row * kPDsRowStride + lane_row]);
                    k_fragment[j] = k_lds[KVLdsElementOffset(kv_row, d)];
                }
                WmmaInPlace(dq_partial, ds_fragment, k_fragment);
            }

#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                const int32_t row = i * 2 + lane_group;
                if(q_start + row < n_q)
                {
                    const int32_t d =
                        (wave * 2 + d_subtile) * 16 + lane_row;
                    atomicAdd(
                        dq_acc + q_head_offset + (q_start + row) * D + d,
                        dq_partial[i] * scale);
                }
            }
        }
        __syncthreads();
    }
}

template <int D>
__global__ void ConvertDqKernel(
    const float* __restrict__ dq_acc,
    __half* __restrict__ dq,
    int32_t elements)
{
    static_assert(D == kFusedHeadDim);
    const int32_t linear =
        static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear < elements)
        dq[linear] = __float2half(dq_acc[linear]);
}

template <int D>
__global__ void ConvertDkDvKernel(
    const float* __restrict__ dk_acc,
    const float* __restrict__ dv_acc,
    __half* __restrict__ dk,
    __half* __restrict__ dv,
    int32_t elements)
{
    static_assert(D == kFusedHeadDim);
    const int32_t linear =
        static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(linear < elements)
    {
        dk[linear] = __float2half(dk_acc[linear]);
        dv[linear] = __float2half(dv_acc[linear]);
    }
}

template <int D>
bool LaunchFusedBackward(const BackwardLaunchParams& params)
{
    static_assert(D == kFusedHeadDim);
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
    auto* dq_acc = reinterpret_cast<float*>(params.dq_acc_ptr);
    auto* dk_acc = reinterpret_cast<float*>(params.dk_acc_ptr);
    auto* dv_acc = reinterpret_cast<float*>(params.dv_acc_ptr);
    const int32_t q_rows = params.head_count * params.n_q;
    const int32_t kv_rows = params.head_count * params.n_kv;

    (void)hipGetLastError();
    LaunchBackwardRows(
        FusedDeltaClearKernel<D>, q_rows, params.stream,
        out, dout, delta, dq_acc, q_rows);

    const int32_t kv_tiles =
        (params.n_kv + kFusedBlockKV - 1) / kFusedBlockKV;
    const dim3 main_block(kFusedBlockSize);
    const dim3 main_grid(
        static_cast<uint32_t>(params.head_count * kv_tiles));
    hipLaunchKernelGGL(
        FusedBackwardKernel<D>,
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
        dq_acc,
        dk_acc,
        dv_acc,
        params.n_q,
        params.n_kv,
        params.scale);

    const int32_t q_elements = q_rows * D;
    const int32_t kv_elements = kv_rows * D;
    LaunchBackwardRows(
        ConvertDqKernel<D>, q_elements, params.stream,
        dq_acc, dq, q_elements);
    LaunchBackwardRows(
        ConvertDkDvKernel<D>, kv_elements, params.stream,
        dk_acc, dv_acc, dk, dv, kv_elements);
    return hipPeekAtLastError() == hipSuccess;
}

} // namespace feather_attn
