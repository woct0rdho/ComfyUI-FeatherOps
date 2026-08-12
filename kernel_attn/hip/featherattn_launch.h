#pragma once

#include <cstdint>

namespace feather_attn {

struct LaunchParams
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void* o_ptr;
    int32_t n_q;
    int32_t n_kv;
    uint32_t grid_size;
    hipStream_t stream;
};

extern "C" bool feather_attn_d64_aligned(const LaunchParams& params);
extern "C" bool feather_attn_d128_aligned(const LaunchParams& params);
extern "C" bool feather_attn_d64_query_tail(const LaunchParams& params);
extern "C" bool feather_attn_d128_query_tail(const LaunchParams& params);
extern "C" bool feather_attn_d64_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_d128_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_d64_query_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_d128_query_key_tail(const LaunchParams& params);

} // namespace feather_attn
