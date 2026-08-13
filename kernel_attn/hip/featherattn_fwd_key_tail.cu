#include "featherattn_fwd_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_hnd_d64_key_tail(const LaunchParams& params)
{
    return LaunchVariant<64, false, false, true>(params);
}

extern "C" bool feather_attn_hnd_d128_key_tail(const LaunchParams& params)
{
    return LaunchVariant<128, false, false, true>(params);
}

extern "C" bool feather_attn_nhd_d64_key_tail(const LaunchParams& params)
{
    return LaunchVariant<64, true, false, true>(params);
}

extern "C" bool feather_attn_nhd_d128_key_tail(const LaunchParams& params)
{
    return LaunchVariant<128, true, false, true>(params);
}

} // namespace feather_attn
