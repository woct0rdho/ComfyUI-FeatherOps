#include "featherattn_bwd_kernel.h"

namespace feather_attn {

extern "C" bool feather_attn_bwd_d128_reference(
    const BackwardLaunchParams& params)
{
    return LaunchReferenceBackward<128>(params);
}

} // namespace feather_attn
