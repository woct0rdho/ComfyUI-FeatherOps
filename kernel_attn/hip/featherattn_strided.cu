#define FEATHER_ATTN_STRIDED_NHD 1
#define AttentionKernel StridedNhdAttentionKernel
#define LaunchVariant LaunchStridedNhdVariant
#include "featherattn_kernel.h"
#undef LaunchVariant
#undef AttentionKernel
#undef FEATHER_ATTN_STRIDED_NHD

namespace feather_attn {

extern "C" bool feather_attn_nhd_d64_strided_aligned(
    const StridedLaunchParams& params)
{
    return LaunchStridedNhdVariant<64, true, false, false>(params);
}

extern "C" bool feather_attn_nhd_d64_strided_query_tail(
    const StridedLaunchParams& params)
{
    return LaunchStridedNhdVariant<64, true, true, false>(params);
}

extern "C" bool feather_attn_nhd_d64_strided_key_tail(
    const StridedLaunchParams& params)
{
    return LaunchStridedNhdVariant<64, true, false, true>(params);
}

extern "C" bool feather_attn_nhd_d64_strided_query_key_tail(
    const StridedLaunchParams& params)
{
    return LaunchStridedNhdVariant<64, true, true, true>(params);
}

} // namespace feather_attn
