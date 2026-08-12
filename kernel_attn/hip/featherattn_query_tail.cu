#include "featherattn_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_d64_query_tail(const LaunchParams& params)
{
    return LaunchVariant<64, true, false>(params);
}

extern "C" bool feather_attn_d128_query_tail(const LaunchParams& params)
{
    return LaunchVariant<128, true, false>(params);
}

} // namespace feather_attn
