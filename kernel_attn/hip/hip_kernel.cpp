#include <hip/hip_runtime.h>

#include "featherattn_launch.h"

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
#include <cstdint>

#ifndef NO_PYTORCH
namespace {

using half_bits_t = uint16_t;

bool IsContiguous4D(const torch::stable::Tensor& tensor)
{
    __int128 expected_stride = 1;
    for(int64_t dim = 3; dim >= 0; --dim)
    {
        if(expected_stride > INT64_MAX ||
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

Launcher SelectLauncher(int64_t head_dim, bool pad_q, bool pad_kv)
{
    if(head_dim == 64)
    {
        if(pad_q && pad_kv)
            return feather_attn::feather_attn_d64_query_key_tail;
        if(pad_q)
            return feather_attn::feather_attn_d64_query_tail;
        if(pad_kv)
            return feather_attn::feather_attn_d64_key_tail;
        return feather_attn::feather_attn_d64_aligned;
    }
    if(pad_q && pad_kv)
        return feather_attn::feather_attn_d128_query_key_tail;
    if(pad_q)
        return feather_attn::feather_attn_d128_query_tail;
    if(pad_kv)
        return feather_attn::feather_attn_d128_key_tail;
    return feather_attn::feather_attn_d128_aligned;
}

} // namespace

void AttnFp16Feather(
    const torch::stable::Tensor& q,
    const torch::stable::Tensor& k,
    const torch::stable::Tensor& v,
    torch::stable::Tensor& o)
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

    const int64_t batch = q.size(0);
    const int64_t heads = q.size(1);
    const int64_t n_q   = q.size(2);
    const int64_t d     = q.size(3);
    const int64_t n_kv  = k.size(2);
    STD_TORCH_CHECK(batch > 0 && heads > 0 && n_q > 0 && n_kv > 0,
                    "batch, heads, N_Q, and N_KV must be positive");
    STD_TORCH_CHECK(d == 64 || d == 128, "head dimension must be 64 or 128");
    STD_TORCH_CHECK(k.size(0) == batch && v.size(0) == batch && o.size(0) == batch,
                    "batch size mismatch");
    STD_TORCH_CHECK(k.size(1) == heads && v.size(1) == heads && o.size(1) == heads,
                    "head count mismatch");
    STD_TORCH_CHECK(k.size(3) == d && v.size(3) == d && o.size(3) == d,
                    "head dimension mismatch");
    STD_TORCH_CHECK(v.size(2) == n_kv && o.size(2) == n_q,
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
    const feather_attn::LaunchParams params{
        q.const_data_ptr(),
        k.const_data_ptr(),
        v.const_data_ptr(),
        o.mutable_data_ptr(),
        static_cast<int32_t>(n_q),
        static_cast<int32_t>(n_kv),
        static_cast<uint32_t>(grid_size),
        stream};

    (void)hipGetLastError();
    const bool launched = SelectLauncher(d, n_q % 128 != 0, n_kv % 64 != 0)(params);
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
        "Tensor(a!) out"
        ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(feather_attn_fp16, CUDA, m)
{
    m.impl("attn_fp16_feather", TORCH_BOX(&AttnFp16Feather));
}
#endif
