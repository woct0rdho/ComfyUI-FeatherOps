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
    int32_t num_heads;
    int32_t head_start;
    int32_t launch_heads;
    uint32_t grid_size;
    hipStream_t stream;
};

struct StridedLaunchParams
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void* o_ptr;
    int32_t n_q;
    int32_t n_kv;
    int32_t num_heads;
    int32_t group_index;
    int32_t launch_heads;
    int32_t group_count;
    uint32_t grid_size;
    hipStream_t stream;
};

extern "C" bool feather_attn_hnd_d64_aligned(const LaunchParams& params);
extern "C" bool feather_attn_hnd_d128_aligned(const LaunchParams& params);
extern "C" bool feather_attn_nhd_d64_aligned(const LaunchParams& params);
extern "C" bool feather_attn_nhd_d128_aligned(const LaunchParams& params);
extern "C" bool feather_attn_hnd_d64_query_tail(const LaunchParams& params);
extern "C" bool feather_attn_hnd_d128_query_tail(const LaunchParams& params);
extern "C" bool feather_attn_nhd_d64_query_tail(const LaunchParams& params);
extern "C" bool feather_attn_nhd_d128_query_tail(const LaunchParams& params);
extern "C" bool feather_attn_hnd_d64_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_hnd_d128_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_nhd_d64_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_nhd_d128_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_hnd_d64_query_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_hnd_d128_query_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_nhd_d64_query_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_nhd_d128_query_key_tail(const LaunchParams& params);
extern "C" bool feather_attn_nhd_d64_strided_aligned(
    const StridedLaunchParams& params);
extern "C" bool feather_attn_nhd_d64_strided_query_tail(
    const StridedLaunchParams& params);
extern "C" bool feather_attn_nhd_d64_strided_key_tail(
    const StridedLaunchParams& params);
extern "C" bool feather_attn_nhd_d64_strided_query_key_tail(
    const StridedLaunchParams& params);

} // namespace feather_attn
