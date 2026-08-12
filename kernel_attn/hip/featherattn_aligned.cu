#include "featherattn_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_hnd_d64_aligned(const LaunchParams& params)
{
    return LaunchVariant<64, false, false, false>(params);
}

extern "C" bool feather_attn_hnd_d128_aligned(const LaunchParams& params)
{
    return LaunchVariant<128, false, false, false>(params);
}

extern "C" bool feather_attn_nhd_d64_aligned(const LaunchParams& params)
{
    return LaunchVariant<64, true, false, false>(params);
}

extern "C" bool feather_attn_nhd_d128_aligned(const LaunchParams& params)
{
    return LaunchVariant<128, true, false, false>(params);
}

} // namespace feather_attn
