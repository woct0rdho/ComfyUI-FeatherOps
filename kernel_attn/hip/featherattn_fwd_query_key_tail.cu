#include "featherattn_fwd_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_hnd_d64_query_key_tail(const LaunchParams& params)
{
    return LaunchVariant<64, false, true, true>(params);
}

extern "C" bool feather_attn_hnd_d128_query_key_tail(const LaunchParams& params)
{
    return LaunchVariant<128, false, true, true>(params);
}

extern "C" bool feather_attn_nhd_d64_query_key_tail(const LaunchParams& params)
{
    return LaunchVariant<64, true, true, true>(params);
}

extern "C" bool feather_attn_nhd_d128_query_key_tail(const LaunchParams& params)
{
    return LaunchVariant<128, true, true, true>(params);
}

} // namespace feather_attn
