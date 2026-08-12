#pragma once

#include <__clang_cuda_math_forward_declares.h>
#include <hip/hip_runtime.h>

#include "featherattn_launch.h"

#if defined(_WIN32)
#include <__clang_hip_math.h>
#endif
#include <ck_tile/core.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_params.hpp>
#include <ck_tile/ops/gemm/warp/warp_wmma_gemm.hpp>

#include <cstdint>

namespace feather_attn {

using ck_tile::fp16_t;
using ck_tile::index_t;
using ck_tile::number;

struct Q8Ops
{
    static constexpr index_t kNumWaves = 8;
    static constexpr index_t kLaneRows = 16;
    static constexpr index_t kDPerWmma = 16;

    template <index_t kDTiles>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptor()
    {
        return ck_tile::make_naive_tensor_descriptor_packed(
            ck_tile::make_tuple(number<kNumWaves>{},
                                number<kDTiles>{},
                                number<kLaneRows>{},
                                number<kDPerWmma>{}));
    }

    CK_TILE_DEVICE static uint8_t EncodeE5M2(fp16_t value)
    {
        const uint16_t bits    = ck_tile::bit_cast<uint16_t>(value);
        const uint16_t rounded = bits + 0x007fu + ((bits >> 8) & 1u);
        return static_cast<uint8_t>(rounded >> 8);
    }

    CK_TILE_DEVICE static ck_tile::fp16x16_t DecodeE5M2x16(const uint8_t* src)
    {
        const auto packed = *reinterpret_cast<const ck_tile::uint32x4_t*>(src);
        ck_tile::uint32x8_t decoded;
        ck_tile::static_for<0, 4, 1>{}([&](auto i) {
            decoded[i.value * 2] =
                __builtin_amdgcn_perm(0u, packed[i.value], 0x010c000cu);
            decoded[i.value * 2 + 1] =
                __builtin_amdgcn_perm(0u, packed[i.value], 0x030c020cu);
        });
        return ck_tile::bit_cast<ck_tile::fp16x16_t>(decoded);
    }
};

struct QKOps
{
    static constexpr index_t kKPack = 8;

    template <index_t kHeadDim>
    CK_TILE_DEVICE static index_t LdsOffset(index_t row, index_t chunk)
    {
        constexpr index_t kKChunks = kHeadDim / kKPack;
        const index_t physical_chunk = chunk ^ (row % kKChunks);
        return (row * kHeadDim + physical_chunk * kKPack) * sizeof(fp16_t);
    }
};

struct SoftmaxOps
{
    using Wmma = ck_tile::WarpGemmAttributeWmmaImpl<
        ck_tile::WmmaTraits<ck_tile::gfx11_t, fp16_t, fp16_t, float, 16, 16, 16>>;

    CK_TILE_DEVICE static float Exp2(float value)
    {
        return __builtin_amdgcn_exp2f(value);
    }

    CK_TILE_DEVICE static typename Wmma::AVecType TransposedCFragToA(
        const typename Wmma::CVecType& fragment,
        index_t lane)
    {
        ck_tile::uint32x8_t words;
        ck_tile::static_for<0, 4, 1>{}([&](auto i) {
            ck_tile::fp16x2_t pair;
            pair[0]          = ck_tile::type_convert<fp16_t>(fragment[i.value * 2]);
            pair[1]          = ck_tile::type_convert<fp16_t>(fragment[i.value * 2 + 1]);
            const uint32_t v = ck_tile::bit_cast<uint32_t>(pair);
            const uint32_t w = __builtin_amdgcn_permlanex16(
                0u, v, 0x76543210u, 0xfedcba98u, false, true);
            const uint32_t selector_0 = lane < 16 ? 0x05040100u : 0x01000504u;
            const uint32_t selector_1 = lane < 16 ? 0x07060302u : 0x03020706u;
            words[i.value * 2]     = __builtin_amdgcn_perm(w, v, selector_0);
            words[i.value * 2 + 1] = __builtin_amdgcn_perm(w, v, selector_1);
        });
        return ck_tile::bit_cast<typename Wmma::AVecType>(words);
    }

    CK_TILE_DEVICE static float SwapLaneGroup(float value)
    {
        const uint32_t bits = __builtin_bit_cast(uint32_t, value);
        return __builtin_bit_cast(
            float,
            __builtin_amdgcn_permlanex16(
                0u, bits, 0x76543210u, 0xfedcba98u, false, true));
    }
};

template <index_t kHeadDimValue, bool kNHD, bool kPadQ, bool kPadKV>
struct AttentionKernel
{
    using Wmma = SoftmaxOps::Wmma;

    static constexpr index_t kBlockSize  = 256;
    static constexpr index_t kBlockM     = 128;
    static constexpr index_t kBlockN     = 64;
    static constexpr index_t kHeadDim    = kHeadDimValue;
    static constexpr index_t kWaveSize   = 32;
    static constexpr index_t kLaneRows   = 16;
    static constexpr index_t kDPerWmma   = 16;
    static constexpr index_t kDTiles     = kHeadDim / kDPerWmma;
    static constexpr index_t kNTiles     = kBlockN / kLaneRows;
    static constexpr index_t kQLdsBytes  = kBlockM * kHeadDim;
    static constexpr index_t kKVLdsBytes = kBlockN * kHeadDim * sizeof(fp16_t);
    static constexpr index_t kLdsBytes   = kQLdsBytes + kKVLdsBytes;
    static constexpr index_t kKPack      = 8;
    static constexpr index_t kKChunks    = kHeadDim / kKPack;
    static constexpr index_t kVPack      = 8;
    static constexpr index_t kVChunks    = kBlockN / kVPack;

    struct Kargs
    {
        const fp16_t* q_ptr;
        const fp16_t* k_ptr;
        const fp16_t* v_ptr;
        fp16_t* o_ptr;
        float q_scale_log2;
        int32_t n_q;
        int32_t n_kv;
        int32_t num_heads;
        int32_t head_start;
        int32_t launch_heads;
    };

    CK_TILE_DEVICE static index_t VLdsOffset(index_t d_row, index_t n_chunk)
    {
        const index_t d_phase = (d_row ^ (d_row / kVChunks)) % kVChunks;
        const index_t physical_chunk = n_chunk ^ d_phase;
        return (d_row * kBlockN + physical_chunk * kVPack) * sizeof(fp16_t);
    }

    CK_TILE_DEVICE static index_t VLdsElementOffset(index_t d_row, index_t n)
    {
        const index_t n_chunk = n / kVPack;
        const index_t d_phase = (d_row ^ (d_row / kVChunks)) % kVChunks;
        const index_t physical_chunk = n_chunk ^ d_phase;
        return (d_row * kBlockN + physical_chunk * kVPack + n % kVPack) *
               sizeof(fp16_t);
    }

    CK_TILE_DEVICE static void WmmaInPlace(
        typename Wmma::CVecType& c,
        const typename Wmma::AVecType& a,
        const typename Wmma::BVecType& b)
    {
        asm volatile(
            "v_wmma_f32_16x16x16_f16 %0, %1, %2, %0"
            : "+v"(c)
            : "v"(a), "v"(b));
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        constexpr auto q_lds_desc = Q8Ops::MakeLdsDescriptor<kDTiles>();
        alignas(16) __shared__ uint8_t lds[kLdsBytes];
        uint8_t* const q_lds  = lds;
        uint8_t* const kv_lds = lds + kQLdsBytes;

        const index_t tid        = ck_tile::get_thread_local_1d_id();
        const index_t wave       = tid / kWaveSize;
        const index_t lane       = tid % kWaveSize;
        const index_t lane_row   = lane % kLaneRows;
        const index_t lane_group = lane / kLaneRows;
        const index_t q_row      = wave * kLaneRows + lane_row;
        index_t q_tiles          = kargs.n_q / kBlockM;
        if constexpr(kPadQ)
            q_tiles += kargs.n_q % kBlockM != 0;
        const index_t block = ck_tile::get_block_1d_id();
        index_t q_tile;
        index_t q_head_offset;
        index_t kv_head_offset;
        if constexpr(kNHD)
        {
            const index_t local_head = block % kargs.launch_heads;
            const index_t head       = kargs.head_start + local_head;
            const index_t tile_batch = block / kargs.launch_heads;
            q_tile                   = tile_batch % q_tiles;
            const index_t batch      = tile_batch / q_tiles;
            q_head_offset =
                (batch * kargs.n_q * kargs.num_heads + head) * kHeadDim;
            kv_head_offset =
                (batch * kargs.n_kv * kargs.num_heads + head) * kHeadDim;
        }
        else
        {
            q_tile                    = block % q_tiles;
            const index_t head_linear = block / q_tiles;
            q_head_offset  = head_linear * kargs.n_q * kHeadDim;
            kv_head_offset = head_linear * kargs.n_kv * kHeadDim;
        }
        const index_t q_start = q_tile * kBlockM;
        const index_t q_row_stride = kNHD ? kargs.num_heads * kHeadDim : kHeadDim;
        const index_t kv_row_stride = q_row_stride;
        const fp16_t* const q_tile_ptr =
            kargs.q_ptr + q_head_offset + q_start * q_row_stride;
        const fp16_t* const k_head_ptr = kargs.k_ptr + kv_head_offset;
        const fp16_t* const v_head_ptr = kargs.v_ptr + kv_head_offset;
        fp16_t* const o_tile_ptr =
            kargs.o_ptr + q_head_offset + q_start * q_row_stride;

        ck_tile::static_for<0, kDTiles / 2, 1>{}([&](auto i) {
            constexpr index_t d_tile_in_group = i.value * 2;
            const index_t d_tile              = d_tile_in_group + lane_group;
            ck_tile::fp16x16_t q = {};
            if constexpr(kPadQ)
            {
                if(q_start + q_row < kargs.n_q)
                    q = *reinterpret_cast<const ck_tile::fp16x16_t*>(
                        q_tile_ptr + q_row * q_row_stride + d_tile * kDPerWmma);
            }
            else
            {
                q = *reinterpret_cast<const ck_tile::fp16x16_t*>(
                    q_tile_ptr + q_row * q_row_stride + d_tile * kDPerWmma);
            }

            ck_tile::uint8x16_t q8;
            ck_tile::static_for<0, kDPerWmma, 1>{}([&](auto j) {
                const float scaled =
                    ck_tile::type_convert<float>(q[j.value]) * kargs.q_scale_log2;
                q8[j.value] =
                    Q8Ops::EncodeE5M2(ck_tile::type_convert<fp16_t>(scaled));
            });

            const index_t q_lds_offset = q_lds_desc.calculate_offset(
                ck_tile::make_tuple(wave, d_tile, lane_row, index_t{0}));
            *reinterpret_cast<ck_tile::uint8x16_t*>(q_lds + q_lds_offset) = q8;
        });

        typename Wmma::CVecType output[kDTiles] = {};
        uint32_t store_wave;
        asm volatile("v_readfirstlane_b32 %0, %1" : "=s"(store_wave) : "v"(wave));
        float lane_lse = -__builtin_inff();
        typename Wmma::AVecType cached_q = {};

        #pragma unroll 1
        for(index_t key_start = 0; key_start < kargs.n_kv; key_start += kBlockN)
        {
            constexpr index_t kKLoadIssues = kBlockN * kKChunks / kBlockSize;
            ck_tile::static_for<0, kKLoadIssues, 1>{}([&](auto issue) {
                const index_t linear_chunk = tid + issue.value * kBlockSize;
                const index_t k_row        = linear_chunk / kKChunks;
                const index_t k_chunk      = linear_chunk % kKChunks;
                ck_tile::fp16x8_t k = {};
                if constexpr(kPadKV)
                {
                    if(key_start + k_row < kargs.n_kv)
                        k = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                            k_head_ptr + (key_start + k_row) * kv_row_stride +
                            k_chunk * kKPack);
                }
                else
                {
                    k = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        k_head_ptr + (key_start + k_row) * kv_row_stride +
                        k_chunk * kKPack);
                }
                *reinterpret_cast<ck_tile::fp16x8_t*>(
                    kv_lds + QKOps::LdsOffset<kHeadDim>(k_row, k_chunk)) = k;
            });

            ck_tile::block_sync_lds();

            typename Wmma::CVecType scores[kNTiles] = {};
            ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
                const index_t q_lds_offset = q_lds_desc.calculate_offset(
                    ck_tile::make_tuple(wave, d_tile.value, lane_row, index_t{0}));
                typename Wmma::AVecType q;
                if constexpr(kHeadDim == 64 && !kNHD && d_tile.value == 0)
                {
                    if(key_start == 0)
                        cached_q = Q8Ops::DecodeE5M2x16(q_lds + q_lds_offset);
                    q = cached_q;
                }
                else
                {
                    q = Q8Ops::DecodeE5M2x16(q_lds + q_lds_offset);
                }

                ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                    const index_t k_row   = n_tile.value * kLaneRows + lane_row;
                    const index_t chunk_0 = d_tile.value * 2;
                    const auto k_lo = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        kv_lds + QKOps::LdsOffset<kHeadDim>(k_row, chunk_0));
                    const auto k_hi = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        kv_lds + QKOps::LdsOffset<kHeadDim>(k_row, chunk_0 + 1));
                    typename Wmma::BVecType k;
                    ck_tile::static_for<0, 8, 1>{}([&](auto j) {
                        k[j.value]     = k_lo[j.value];
                        k[j.value + 8] = k_hi[j.value];
                    });
                    WmmaInPlace(scores[n_tile.value], k, q);
                });
            });

            if constexpr(kPadKV)
            {
                ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                        const index_t key_row = key_start + n_tile.value * kLaneRows +
                                                i.value * 2 + lane_group;
                        if(key_row >= kargs.n_kv)
                            scores[n_tile.value][i.value] = -__builtin_inff();
                    });
                });
            }

            float row_max = -__builtin_inff();
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                    row_max = __builtin_fmaxf(row_max, scores[n_tile.value][i.value]);
                });
            });
            row_max = __builtin_fmaxf(row_max, SoftmaxOps::SwapLaneGroup(row_max));

            float local_sum = 0.0f;
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                    const float p =
                        SoftmaxOps::Exp2(scores[n_tile.value][i.value] - row_max);
                    scores[n_tile.value][i.value] = p;
                    local_sum += p;
                });
            });
            const float tile_sum  = local_sum + SoftmaxOps::SwapLaneGroup(local_sum);
            const float old_lse   = lane_lse;
            const float new_max   = __builtin_fmaxf(old_lse, row_max);
            const float old_term  = SoftmaxOps::Exp2(old_lse - new_max);
            const float tile_term = SoftmaxOps::Exp2(row_max - new_max);
            const float combined  = old_term + tile_term * tile_sum;
            const float reciprocal = __builtin_amdgcn_rcpf(combined);
            const float alpha      = old_term * reciprocal;
            const float beta       = tile_term * reciprocal;
            lane_lse               = new_max + __builtin_amdgcn_logf(combined);

            const int alpha_bits = __builtin_bit_cast(int, alpha);
            ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                const int even_alpha_bits =
                    __builtin_amdgcn_readlane(alpha_bits, i.value * 2);
                const int odd_alpha_bits =
                    __builtin_amdgcn_readlane(alpha_bits, i.value * 2 + 1);
                const float row_alpha = __builtin_bit_cast(
                    float, lane_group == 0 ? even_alpha_bits : odd_alpha_bits);
                ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
                    output[d_tile.value][i.value] *= row_alpha;
                });
            });
            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                    scores[n_tile.value][i.value] *= beta;
                });
            });

            ck_tile::block_sync_lds();

            index_t kv_lane_row;
            asm volatile("v_mov_b32 %0, %1" : "=v"(kv_lane_row) : "v"(lane_row));
            constexpr index_t kVDChunks = kHeadDim / kVPack;
            constexpr index_t kVDGroups = kLaneRows / kVDChunks;
            constexpr index_t kVRowsPerLane = 4 / kVDGroups;
            constexpr index_t kVColsPerLane = 8;
            const index_t v_n_base = wave * 8 + lane_group * (kVRowsPerLane * kVDGroups) +
                                     (lane_row / kVDChunks) * kVRowsPerLane;
            const index_t v_d_base = (lane_row % kVDChunks) * kVColsPerLane;
            ck_tile::fp16x8_t v_rows[kVRowsPerLane] = {};
            ck_tile::static_for<0, kVRowsPerLane, 1>{}([&](auto row) {
                if constexpr(kPadKV)
                {
                    if(key_start + v_n_base + row.value < kargs.n_kv)
                        v_rows[row.value] = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                            v_head_ptr +
                            (key_start + v_n_base + row.value) * kv_row_stride +
                            v_d_base);
                }
                else
                {
                    v_rows[row.value] = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        v_head_ptr +
                        (key_start + v_n_base + row.value) * kv_row_stride +
                        v_d_base);
                }
            });
            ck_tile::static_for<0, kVColsPerLane, 1>{}([&](auto col) {
                if constexpr(kVRowsPerLane == 4)
                {
                    ck_tile::fp16x4_t v_column;
                    ck_tile::static_for<0, kVRowsPerLane, 1>{}([&](auto row) {
                        v_column[row.value] = v_rows[row.value][col.value];
                    });
                    *reinterpret_cast<ck_tile::fp16x4_t*>(
                        kv_lds + VLdsElementOffset(v_d_base + col.value, v_n_base)) =
                        v_column;
                }
                else
                {
                    ck_tile::fp16x2_t v_column;
                    ck_tile::static_for<0, kVRowsPerLane, 1>{}([&](auto row) {
                        v_column[row.value] = v_rows[row.value][col.value];
                    });
                    *reinterpret_cast<ck_tile::fp16x2_t*>(
                        kv_lds + VLdsElementOffset(v_d_base + col.value, v_n_base)) =
                        v_column;
                }
            });

            ck_tile::block_sync_lds();

            ck_tile::static_for<0, kNTiles, 1>{}([&](auto n_tile) {
                const typename Wmma::AVecType p =
                    SoftmaxOps::TransposedCFragToA(scores[n_tile.value], lane);
                ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
                    const index_t d_row = d_tile.value * kDPerWmma + kv_lane_row;
                    const auto v_lo = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        kv_lds + VLdsOffset(d_row, n_tile.value * 2));
                    const auto v_hi = *reinterpret_cast<const ck_tile::fp16x8_t*>(
                        kv_lds + VLdsOffset(d_row, n_tile.value * 2 + 1));
                    typename Wmma::BVecType v;
                    ck_tile::static_for<0, 8, 1>{}([&](auto j) {
                        v[j.value]     = v_lo[j.value];
                        v[j.value + 8] = v_hi[j.value];
                    });
                    WmmaInPlace(output[d_tile.value], p, v);
                });
            });

            ck_tile::block_sync_lds();
        }

        uint32_t store_lane;
        asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(store_lane));
        ck_tile::static_for<0, kDTiles, 1>{}([&](auto d_tile) {
            ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                const index_t row =
                    store_wave * kLaneRows + i.value * 2 + store_lane / kLaneRows;
                const index_t col =
                    d_tile.value * kDPerWmma + store_lane % kLaneRows;
                if constexpr(kPadQ)
                {
                    if(q_start + row < kargs.n_q)
                        o_tile_ptr[row * q_row_stride + col] =
                            ck_tile::type_convert<fp16_t>(output[d_tile.value][i.value]);
                }
                else
                {
                    o_tile_ptr[row * q_row_stride + col] =
                        ck_tile::type_convert<fp16_t>(output[d_tile.value][i.value]);
                }
            });
        });
    }
};

template <index_t kHeadDim, bool kNHD, bool kPadQ, bool kPadKV>
bool LaunchVariant(const LaunchParams& params)
{
    static_assert(kHeadDim == 64 || kHeadDim == 128);
    using Kernel = AttentionKernel<kHeadDim, kNHD, kPadQ, kPadKV>;
    constexpr float q_scale_log2 =
        kHeadDim == 64 ? 0.18033688011112042f : 0.12751743082459868f;
    const typename Kernel::Kargs kargs{
        reinterpret_cast<const fp16_t*>(params.q_ptr),
        reinterpret_cast<const fp16_t*>(params.k_ptr),
        reinterpret_cast<const fp16_t*>(params.v_ptr),
        reinterpret_cast<fp16_t*>(params.o_ptr),
        q_scale_log2,
        params.n_q,
        params.n_kv,
        params.num_heads,
        params.head_start,
        params.launch_heads};
    const auto kernel = ck_tile::make_kernel<8, ck_tile::gfx115_t>(
        Kernel{},
        dim3(params.grid_size),
        dim3(Kernel::kBlockSize),
        0,
        kargs);
    (void)hipGetLastError();
    ck_tile::launch_kernel(ck_tile::stream_config{params.stream}, kernel);
    return hipPeekAtLastError() == hipSuccess;
}

} // namespace feather_attn
