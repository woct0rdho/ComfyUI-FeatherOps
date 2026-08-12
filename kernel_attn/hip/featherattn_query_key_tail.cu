#include "featherattn_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_d64_query_key_tail(const LaunchParams& params)
{
    return LaunchVariant<64, true, true>(params);
}

extern "C" bool feather_attn_d128_query_key_tail(const LaunchParams& params)
{
    return LaunchVariant<128, true, true>(params);
}

} // namespace feather_attn
