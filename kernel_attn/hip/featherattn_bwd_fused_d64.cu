#include "featherattn_bwd_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_bwd_d64_fused(
    const BackwardLaunchParams& params)
{
    return LaunchFusedBackward<64>(params);
}

} // namespace feather_attn
