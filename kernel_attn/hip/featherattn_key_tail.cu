#include "featherattn_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_d64_key_tail(const LaunchParams& params)
{
    return LaunchVariant<64, false, true>(params);
}

extern "C" bool feather_attn_d128_key_tail(const LaunchParams& params)
{
    return LaunchVariant<128, false, true>(params);
}

} // namespace feather_attn
