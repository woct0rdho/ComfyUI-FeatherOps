#include "featherattn_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_hnd_d64_query_tail(const LaunchParams& params)
{
    return LaunchVariant<64, false, true, false>(params);
}

extern "C" bool feather_attn_hnd_d128_query_tail(const LaunchParams& params)
{
    return LaunchVariant<128, false, true, false>(params);
}

extern "C" bool feather_attn_nhd_d64_query_tail(const LaunchParams& params)
{
    return LaunchVariant<64, true, true, false>(params);
}

extern "C" bool feather_attn_nhd_d128_query_tail(const LaunchParams& params)
{
    return LaunchVariant<128, true, true, false>(params);
}

} // namespace feather_attn
