#include "featherattn_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_d64_aligned(const LaunchParams& params)
{
    return LaunchVariant<64, false, false>(params);
}

extern "C" bool feather_attn_d128_aligned(const LaunchParams& params)
{
    return LaunchVariant<128, false, false>(params);
}

} // namespace feather_attn
