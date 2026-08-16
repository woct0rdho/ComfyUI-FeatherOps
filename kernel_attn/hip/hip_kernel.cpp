#include <hip/hip_runtime.h>

#include "hip_kernel.h"

#ifndef NO_PYTORCH
#if defined(__clang__)
// Thrust emits a gfx1100/gfx1101 clock warning for gfx1151. This code does not use timing APIs.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#endif
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif

#include <climits>
#include <cmath>
#include <cstdint>

#ifndef NO_PYTORCH
namespace {

using half_bits_t = uint16_t;

bool IsContiguous4D(const torch::stable::Tensor& tensor)
{
    __int128 expected_stride = 1;
    for(int64_t dim = 3; dim >= 0; --dim)
    {
        if(expected_stride > INT64_MAX)
            return false;
        if(tensor.size(dim) != 1 &&
           tensor.stride(dim) != static_cast<int64_t>(expected_stride))
            return false;
        expected_stride *= tensor.size(dim);
    }
    return true;
}

bool IsInt32HalfAddressable(const torch::stable::Tensor& tensor)
{
    __int128 max_element_offset = 0;
    for(int64_t dim = 0; dim < 4; ++dim)
    {
        const int64_t stride = tensor.stride(dim);
        if(stride < 0 || stride > INT32_MAX)
            return false;
        max_element_offset +=
            static_cast<__int128>(tensor.size(dim) - 1) * stride;
    }
    return max_element_offset * static_cast<int64_t>(sizeof(half_bits_t)) <=
           INT32_MAX;
}

using Launcher = bool (*)(const feather_attn::LaunchParams&);
using StridedLauncher =
    bool (*)(const feather_attn::StridedLaunchParams&);

Launcher SelectLauncher(int64_t head_dim, bool nhd, bool pad_q, bool pad_kv)
{
    if(head_dim == 64)
    {
        if(pad_q && pad_kv)
            return nhd ? feather_attn::feather_attn_nhd_d64_query_key_tail
                       : feather_attn::feather_attn_hnd_d64_query_key_tail;
        if(pad_q)
            return nhd ? feather_attn::feather_attn_nhd_d64_query_tail
                       : feather_attn::feather_attn_hnd_d64_query_tail;
        if(pad_kv)
            return nhd ? feather_attn::feather_attn_nhd_d64_key_tail
                       : feather_attn::feather_attn_hnd_d64_key_tail;
        return nhd ? feather_attn::feather_attn_nhd_d64_aligned
                   : feather_attn::feather_attn_hnd_d64_aligned;
    }
    if(pad_q && pad_kv)
        return nhd ? feather_attn::feather_attn_nhd_d128_query_key_tail
                   : feather_attn::feather_attn_hnd_d128_query_key_tail;
    if(pad_q)
        return nhd ? feather_attn::feather_attn_nhd_d128_query_tail
                   : feather_attn::feather_attn_hnd_d128_query_tail;
    if(pad_kv)
        return nhd ? feather_attn::feather_attn_nhd_d128_key_tail
                   : feather_attn::feather_attn_hnd_d128_key_tail;
    return nhd ? feather_attn::feather_attn_nhd_d128_aligned
               : feather_attn::feather_attn_hnd_d128_aligned;
}

int64_t ResolveNhdHeadGroupSize(int64_t heads, int64_t n_kv, int64_t head_dim)
{
    if(head_dim != 128)
        return heads;

    constexpr __int128 kLlcBytes = static_cast<__int128>(32) * 1024 * 1024;
    const __int128 kv_bytes_per_head =
        static_cast<__int128>(n_kv) * head_dim * 2 * sizeof(half_bits_t);
    const __int128 total_kv_bytes = static_cast<__int128>(heads) * kv_bytes_per_head;
    if(total_kv_bytes * 2 < kLlcBytes * 3 || kv_bytes_per_head > kLlcBytes)
        return heads;

    const int64_t group_size = static_cast<int64_t>(kLlcBytes / kv_bytes_per_head);
    if(group_size < 4 || group_size >= heads)
        return heads;
    return group_size;
}

int64_t ResolveNhdD64StridedGroupCount(int64_t heads, int64_t n_kv)
{
    if(heads % 16 != 0)
        return 1;

    constexpr __int128 kLlcBytes = static_cast<__int128>(32) * 1024 * 1024;
    constexpr int64_t kHeadDim  = 64;
    const __int128 kv_bytes_per_head =
        static_cast<__int128>(n_kv) * kHeadDim * 2 * sizeof(half_bits_t);
    const __int128 total_kv_bytes =
        static_cast<__int128>(heads) * kv_bytes_per_head;
    if(total_kv_bytes * 2 < kLlcBytes * 3 || kv_bytes_per_head > kLlcBytes)
        return 1;

    const int64_t group_size = static_cast<int64_t>(kLlcBytes / kv_bytes_per_head);
    if(group_size < 4 || group_size >= heads)
        return 1;

    int64_t group_count = (heads + group_size - 1) / group_size;
    if(group_count < 3)
        return 1;
    // A physical-head stride divisible by eight recreates the NHD partition cliff.
    if(group_count % 8 == 0)
        ++group_count;
    return group_count;
}

StridedLauncher SelectNhdD64StridedLauncher(bool pad_q, bool pad_kv)
{
    if(pad_q && pad_kv)
        return feather_attn::feather_attn_nhd_d64_strided_query_key_tail;
    if(pad_q)
        return feather_attn::feather_attn_nhd_d64_strided_query_tail;
    if(pad_kv)
        return feather_attn::feather_attn_nhd_d64_strided_key_tail;
    return feather_attn::feather_attn_nhd_d64_strided_aligned;
}

} // namespace

void AttnFp16Feather(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    torch::stable::Tensor& o,
    int64_t layout)
{
    STD_TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && o.is_cuda(),
                    "q, k, v, and o must be CUDA tensors");
    const auto device_index = q.get_device_index();
    STD_TORCH_CHECK(k.get_device_index() == device_index &&
                        v.get_device_index() == device_index &&
                        o.get_device_index() == device_index,
                    "q, k, v, and o must be on the same device");
    STD_TORCH_CHECK(q.scalar_type() == torch::stable::ScalarType::Half &&
                        k.scalar_type() == torch::stable::ScalarType::Half &&
                        v.scalar_type() == torch::stable::ScalarType::Half &&
                        o.scalar_type() == torch::stable::ScalarType::Half,
                    "q, k, v, and o must be float16");
    STD_TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && o.dim() == 4,
                    "q, k, v, and o must be 4D");

    STD_TORCH_CHECK(layout == 0 || layout == 1,
                    "layout must be 0 (HND) or 1 (NHD)");
    const bool nhd      = layout == 1;
    const int64_t batch = q.size(0);
    const int64_t heads = q.size(nhd ? 2 : 1);
    const int64_t n_q   = q.size(nhd ? 1 : 2);
    const int64_t d     = q.size(3);
    const int64_t n_kv  = k.size(nhd ? 1 : 2);
    STD_TORCH_CHECK(batch > 0 && heads > 0 && n_q > 0 && n_kv > 0,
                    "batch, heads, N_Q, and N_KV must be positive");
    STD_TORCH_CHECK(d == 64 || d == 128, "head dimension must be 64 or 128");
    STD_TORCH_CHECK(k.size(0) == batch && v.size(0) == batch && o.size(0) == batch,
                    "batch size mismatch");
    const int64_t head_axis = nhd ? 2 : 1;
    const int64_t seq_axis  = nhd ? 1 : 2;
    STD_TORCH_CHECK(k.size(head_axis) == heads &&
                        v.size(head_axis) == heads &&
                        o.size(head_axis) == heads,
                    "head count mismatch");
    STD_TORCH_CHECK(k.size(3) == d && v.size(3) == d && o.size(3) == d,
                    "head dimension mismatch");
    STD_TORCH_CHECK(v.size(seq_axis) == n_kv && o.size(seq_axis) == n_q,
                    "sequence length mismatch");
    STD_TORCH_CHECK(IsContiguous4D(q) && IsContiguous4D(k) &&
                        IsContiguous4D(v) && IsContiguous4D(o),
                    "q, k, v, and o must be contiguous");
    STD_TORCH_CHECK(IsInt32HalfAddressable(q) && IsInt32HalfAddressable(k) &&
                        IsInt32HalfAddressable(v) && IsInt32HalfAddressable(o),
                    "q, k, v, or o exceeds the signed-int32 byte-offset contract");
    STD_TORCH_CHECK(batch <= INT32_MAX && heads <= INT32_MAX &&
                        n_q <= INT32_MAX && n_kv <= INT32_MAX,
                    "batch, heads, N_Q, or N_KV exceeds signed int32");

    const __int128 q_tiles = n_q / 128 + (n_q % 128 != 0);
    const __int128 grid_size = static_cast<__int128>(batch) * heads * q_tiles;
    STD_TORCH_CHECK(grid_size > 0 && grid_size <= UINT32_MAX,
                    "launch grid exceeds unsigned int32");

    torch::stable::accelerator::DeviceGuard device_guard(device_index);
    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const auto stream = reinterpret_cast<hipStream_t>(raw_stream);
    const int64_t group_size = nhd ? ResolveNhdHeadGroupSize(heads, n_kv, d) : heads;
    const int64_t strided_group_count =
        nhd && d == 64 ? ResolveNhdD64StridedGroupCount(heads, n_kv) : 1;
    const bool use_strided_d64 = strided_group_count > 1;
    const Launcher launcher =
        SelectLauncher(d, nhd, n_q % 128 != 0, n_kv % 64 != 0);
    const StridedLauncher strided_launcher =
        use_strided_d64 ? SelectNhdD64StridedLauncher(n_q % 128 != 0, n_kv % 64 != 0)
                        : nullptr;

    (void)hipGetLastError();
    bool launched = true;
    if(use_strided_d64)
    {
        STD_TORCH_CHECK(strided_group_count <= INT32_MAX,
                        "strided head-group count exceeds signed int32");
        for(int64_t group_index = 0; group_index < strided_group_count;
            ++group_index)
        {
            const int64_t launch_heads =
                (heads - 1 - group_index) / strided_group_count + 1;
            const __int128 grouped_grid_size =
                static_cast<__int128>(batch) * launch_heads * q_tiles;
            STD_TORCH_CHECK(launch_heads > 0 && launch_heads <= INT32_MAX &&
                                grouped_grid_size <= UINT32_MAX,
                            "strided head-group launch exceeds device limits");
            const feather_attn::StridedLaunchParams params{
                q.const_data_ptr(),
                k.const_data_ptr(),
                v.const_data_ptr(),
                o.mutable_data_ptr(),
                static_cast<int32_t>(n_q),
                static_cast<int32_t>(n_kv),
                static_cast<int32_t>(heads),
                static_cast<int32_t>(group_index),
                static_cast<int32_t>(launch_heads),
                static_cast<int32_t>(strided_group_count),
                static_cast<uint32_t>(grouped_grid_size),
                stream};
            if(!strided_launcher(params))
            {
                launched = false;
                break;
            }
        }
    }
    else
    {
        for(int64_t head_start = 0; head_start < heads; head_start += group_size)
        {
            const int64_t launch_heads =
                group_size < heads - head_start ? group_size : heads - head_start;
            const __int128 grouped_grid_size =
                static_cast<__int128>(batch) * launch_heads * q_tiles;
            const feather_attn::LaunchParams params{
                q.const_data_ptr(),
                k.const_data_ptr(),
                v.const_data_ptr(),
                o.mutable_data_ptr(),
                static_cast<int32_t>(n_q),
                static_cast<int32_t>(n_kv),
                static_cast<int32_t>(heads),
                static_cast<int32_t>(head_start),
                static_cast<int32_t>(launch_heads),
                static_cast<uint32_t>(grouped_grid_size),
                stream};
            if(!launcher(params))
            {
                launched = false;
                break;
            }
        }
    }
    const hipError_t launch_error = hipGetLastError();
    STD_TORCH_CHECK(launched && launch_error == hipSuccess,
                    "FeatherAttn launch failed: ",
                    hipGetErrorString(launch_error));
}

STABLE_TORCH_LIBRARY(feather_attn_fp16, m)
{
    m.def(
        "attn_fp16_feather("
        "Tensor q, "
        "Tensor k, "
        "Tensor v, "
        "Tensor(a!) out, "
        "int layout"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn_fp16, CUDA, m)
{
    m.impl("attn_fp16_feather", TORCH_BOX(&AttnFp16Feather));
}

namespace feather_attn {
namespace {

bool IsContiguous3D(const torch::stable::Tensor& tensor)
{
    __int128 expected_stride = 1;
    for(int64_t dim = 2; dim >= 0; --dim)
    {
        if(expected_stride > INT64_MAX)
            return false;
        if(tensor.size(dim) != 1 &&
           tensor.stride(dim) != static_cast<int64_t>(expected_stride))
            return false;
        expected_stride *= tensor.size(dim);
    }
    return true;
}

bool IsInt32Addressable(
    const torch::stable::Tensor& tensor,
    int64_t element_size)
{
    __int128 max_element_offset = 0;
    for(int64_t dim = 0; dim < tensor.dim(); ++dim)
    {
        const int64_t stride = tensor.stride(dim);
        if(stride < 0 || stride > INT32_MAX)
            return false;
        max_element_offset +=
            static_cast<__int128>(tensor.size(dim) - 1) * stride;
    }
    return max_element_offset * element_size <= INT32_MAX;
}

} // namespace

void AttnBwdFp16Feather(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    const torch::stable::Tensor& out,
    const torch::stable::Tensor& lse,
    const torch::stable::Tensor& dout,
    torch::stable::Tensor& dq,
    torch::stable::Tensor& dk,
    torch::stable::Tensor& dv,
    torch::stable::Tensor& delta,
    double sm_scale,
    int64_t implementation)
{
    STD_TORCH_CHECK(implementation == 1,
                    "only fused D64 backward is currently supported");
    STD_TORCH_CHECK(q.is_cuda(), "all backward tensors must be CUDA tensors");
    const auto device_index = q.get_device_index();
    auto CheckDevice = [&](const torch::stable::Tensor& tensor) {
        STD_TORCH_CHECK(tensor.is_cuda(), "all backward tensors must be CUDA tensors");
        STD_TORCH_CHECK(tensor.get_device_index() == device_index,
                        "all backward tensors must be on the same device");
    };
    CheckDevice(k);
    CheckDevice(v);
    CheckDevice(out);
    CheckDevice(lse);
    CheckDevice(dout);
    CheckDevice(dq);
    CheckDevice(dk);
    CheckDevice(dv);
    CheckDevice(delta);
    STD_TORCH_CHECK(q.scalar_type() == torch::stable::ScalarType::Half &&
                        k.scalar_type() == torch::stable::ScalarType::Half &&
                        v.scalar_type() == torch::stable::ScalarType::Half &&
                        out.scalar_type() == torch::stable::ScalarType::Half &&
                        dout.scalar_type() == torch::stable::ScalarType::Half &&
                        dq.scalar_type() == torch::stable::ScalarType::Half &&
                        dk.scalar_type() == torch::stable::ScalarType::Half &&
                        dv.scalar_type() == torch::stable::ScalarType::Half,
                    "q, k, v, out, dout, dq, dk, and dv must be float16");
    STD_TORCH_CHECK(lse.scalar_type() == torch::stable::ScalarType::Float &&
                        delta.scalar_type() == torch::stable::ScalarType::Float,
                    "lse and delta must be float32");
    auto CheckAttentionDim = [&](const torch::stable::Tensor& tensor) {
        STD_TORCH_CHECK(tensor.dim() == 4, "attention tensors must be 4D");
    };
    CheckAttentionDim(q);
    CheckAttentionDim(k);
    CheckAttentionDim(v);
    CheckAttentionDim(out);
    CheckAttentionDim(dout);
    CheckAttentionDim(dq);
    CheckAttentionDim(dk);
    CheckAttentionDim(dv);

    const int64_t batch = q.size(0);
    const int64_t heads = q.size(1);
    const int64_t n_q = q.size(2);
    const int64_t head_dim = q.size(3);
    const int64_t n_kv = k.size(2);
    STD_TORCH_CHECK(batch > 0 && heads > 0 && n_q > 0 && n_kv > 0,
                    "attention dimensions must be positive");
    STD_TORCH_CHECK(head_dim == 64,
                    "only D64 fused backward is currently supported");
    STD_TORCH_CHECK(lse.dim() == 3 && delta.dim() == 3,
                    "lse and delta must be [B, H, N]");
    STD_TORCH_CHECK(k.size(0) == batch && v.size(0) == batch &&
                        out.size(0) == batch && dout.size(0) == batch &&
                        dq.size(0) == batch && dk.size(0) == batch &&
                        dv.size(0) == batch,
                    "batch dimensions must match");
    STD_TORCH_CHECK(k.size(1) == heads && v.size(1) == heads &&
                        out.size(1) == heads && dout.size(1) == heads &&
                        dq.size(1) == heads && dk.size(1) == heads &&
                        dv.size(1) == heads,
                    "head dimensions must match");
    STD_TORCH_CHECK(k.size(2) == n_kv && v.size(2) == n_kv &&
                        dk.size(2) == n_kv && dv.size(2) == n_kv,
                    "key/value sequence dimensions must match");
    STD_TORCH_CHECK(out.size(2) == n_q && dout.size(2) == n_q &&
                        dq.size(2) == n_q,
                    "query/output sequence dimensions must match");
    STD_TORCH_CHECK(k.size(3) == head_dim && v.size(3) == head_dim &&
                        out.size(3) == head_dim && dout.size(3) == head_dim &&
                        dq.size(3) == head_dim && dk.size(3) == head_dim &&
                        dv.size(3) == head_dim,
                    "head dimensions must match");
    STD_TORCH_CHECK(lse.size(0) == batch && lse.size(1) == heads &&
                        lse.size(2) == n_q && delta.size(0) == batch &&
                        delta.size(1) == heads && delta.size(2) == n_q,
                    "lse and delta shapes must be [B, H, NQ]");
    auto CheckAttentionLayout = [&](const torch::stable::Tensor& tensor) {
        STD_TORCH_CHECK(tensor.is_contiguous(),
                        "HND backward tensors must be contiguous");
        STD_TORCH_CHECK(IsInt32Addressable(tensor, sizeof(uint16_t)),
                        "an attention tensor exceeds signed-int32 addressing");
    };
    CheckAttentionLayout(q);
    CheckAttentionLayout(k);
    CheckAttentionLayout(v);
    CheckAttentionLayout(out);
    CheckAttentionLayout(dout);
    CheckAttentionLayout(dq);
    CheckAttentionLayout(dk);
    CheckAttentionLayout(dv);
    STD_TORCH_CHECK(IsContiguous3D(lse) && IsContiguous3D(delta),
                    "lse and delta must be contiguous");
    STD_TORCH_CHECK(IsInt32Addressable(lse, sizeof(float)) &&
                        IsInt32Addressable(delta, sizeof(float)),
                    "lse or delta exceeds signed-int32 addressing");
    STD_TORCH_CHECK(batch <= INT32_MAX && heads <= INT32_MAX &&
                        n_q <= INT32_MAX && n_kv <= INT32_MAX,
                    "attention dimensions exceed signed int32");
    STD_TORCH_CHECK(std::isfinite(sm_scale) && sm_scale > 0.0,
                    "sm_scale must be finite and positive");

    const __int128 head_count = static_cast<__int128>(batch) * heads;
    const __int128 q_rows = head_count * n_q;
    const __int128 kv_rows = head_count * n_kv;
    const __int128 q_elements = q_rows * head_dim;
    const __int128 kv_elements = kv_rows * head_dim;
    STD_TORCH_CHECK(head_count <= INT32_MAX && q_rows <= INT32_MAX &&
                        kv_rows <= INT32_MAX && q_elements <= INT32_MAX &&
                        kv_elements <= INT32_MAX,
                    "backward tensor size exceeds signed int32");

    torch::stable::accelerator::DeviceGuard device_guard(device_index);
    void* raw_stream = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
    const BackwardLaunchParams params{
        q.const_data_ptr(),
        k.const_data_ptr(),
        v.const_data_ptr(),
        out.const_data_ptr(),
        lse.const_data_ptr(),
        dout.const_data_ptr(),
        dq.mutable_data_ptr(),
        dk.mutable_data_ptr(),
        dv.mutable_data_ptr(),
        delta.mutable_data_ptr(),
        static_cast<int32_t>(head_count),
        static_cast<int32_t>(n_q),
        static_cast<int32_t>(n_kv),
        static_cast<float>(sm_scale),
        reinterpret_cast<hipStream_t>(raw_stream)};
    (void)hipGetLastError();
    const bool launched = feather_attn_bwd_d64_fused(params);
    const hipError_t launch_error = hipGetLastError();
    STD_TORCH_CHECK(launched && launch_error == hipSuccess,
                    "FeatherAttn backward launch failed: ",
                    hipGetErrorString(launch_error));
}

} // namespace feather_attn

STABLE_TORCH_LIBRARY_FRAGMENT(feather_attn_fp16, m)
{
    m.def(
        "attn_bwd_fp16_feather("
        "Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor dout, "
        "Tensor(a!) dq, Tensor(b!) dk, Tensor(c!) dv, Tensor(d!) delta, "
        "float sm_scale, int implementation"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn_fp16, CUDA, m)
{
    m.impl("attn_bwd_fp16_feather", TORCH_BOX(&feather_attn::AttnBwdFp16Feather));
}
#endif
