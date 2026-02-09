// CK-based mixed-precision GEMM (A=fp16, B=fp8->fp16) with optional scale/bias.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <ck/stream_config.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/sequence.hpp>

namespace ck {
namespace tensor_operation {
namespace element_wise {

template <index_t N>
struct FastNumericArrayConverter<f8_t, half_t, N>
{
    using InputArray  = vector_type<f8_t, N>;
    using OutputArray = vector_type<half_t, N>;

    __device__ static OutputArray convert(InputArray const& Input)
    {
        OutputArray Output;
        static_for<0, N, 1>{}([&](auto i) {
            Output.template AsType<half_t>()(i) =
                type_convert<half_t>(Input.template AsType<f8_t>()[i]);
        });
        return Output;
    }

    __device__ OutputArray operator()(InputArray const& Input) { return convert(Input); }
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck

namespace {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

__device__ __forceinline__ half fp8e4m3fn_to_half(uint8_t x)
{
    const uint16_t x_u16 = static_cast<uint16_t>(x);
    const uint16_t sign = static_cast<uint16_t>((x_u16 & 0x80u) << 8);
    uint16_t exp_mant = static_cast<uint16_t>((x_u16 & 0x7Fu) << 7);
    exp_mant = static_cast<uint16_t>(exp_mant + 0x2000u);
    uint16_t bits = static_cast<uint16_t>(sign | exp_mant);
    if ((x_u16 & 0x78u) == 0u) {
        bits = sign;
    }
    __half_raw r;
    r.x = bits;
    return r;
}

__global__ void fp8_to_half_kernel(const uint8_t* src, half* dst, int64_t count)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = fp8e4m3fn_to_half(src[idx]);
    }
}

template <
    typename Dtype,
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int K1,
    int MPER_WMMA,
    int NPER_WMMA,
    int MPER_WAVE,
    int NPER_WAVE,
    typename ABLOCK_CLUSTER_LENS,
    typename ABLOCK_CLUSTER_ORDER,
    typename ABLOCK_SRC_ORDER,
    int ABLOCK_VECTOR_DIM,
    int ABLOCK_SCALAR_VEC,
    int ABLOCK_SCALAR_VEC_K1,
    bool ABLOCK_LDS_EXTRAM,
    typename BBLOCK_CLUSTER_LENS,
    typename BBLOCK_CLUSTER_ORDER,
    typename BBLOCK_SRC_ORDER,
    int BBLOCK_VECTOR_DIM,
    int BBLOCK_SCALAR_VEC,
    int BBLOCK_SCALAR_VEC_AK1,
    bool BBLOCK_LDS_EXTRAN,
    int CMPER_WAVE,
    int CNPER_WAVE,
    typename CBLOCK_CLUSTER_LENS,
    int CNPER_BLOCK,
    bool PADDING = false,
    bool TRANSA = false,
    bool TRANSB = false>
void gemm_impl_wmma_noswap(CUDABLAS_GEMM_ARGTYPES(Dtype))
{
    const int M = m;
    const int N = n;
    const int K = k;

    const int StrideA = lda;
    const int StrideB = ldb;
    const int StrideC = ldc;

    using ADataType = ck::half_t;
    using BDataType = ck::half_t;
    using CDataType = ck::half_t;
    using AccDataType = float;
    using CShuffleDataType = ck::half_t;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    using ALayout = typename std::conditional<TRANSA, Col, Row>::type;
    using BLayout = typename std::conditional<TRANSB, Col, Row>::type;
    using CLayout = Row;

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;

    static constexpr auto GemmDefault =
        ck::tensor_operation::device::GemmSpecialization::Default;
    static constexpr auto GemmMNKPadding =
        ck::tensor_operation::device::GemmSpecialization::MNKPadding;
    static constexpr auto GemmSpec = PADDING ? GemmMNKPadding : GemmDefault;

    using DeviceGemmInstance =
        ck::tensor_operation::device::DeviceGemmWmma_CShuffle<ALayout,
                                                              BLayout,
                                                              CLayout,
                                                              ADataType,
                                                              BDataType,
                                                              CDataType,
                                                              AccDataType,
                                                              CShuffleDataType,
                                                              AElementOp,
                                                              BElementOp,
                                                              CElementOp,
                                                              GemmSpec,
                                                              1,
                                                              BLOCK_SIZE,
                                                              MBLOCK,
                                                              NBLOCK,
                                                              KBLOCK,
                                                              K1,
                                                              MPER_WMMA,
                                                              NPER_WMMA,
                                                              MPER_WAVE,
                                                              NPER_WAVE,
                                                              ABLOCK_CLUSTER_LENS,
                                                              ABLOCK_CLUSTER_ORDER,
                                                              ABLOCK_SRC_ORDER,
                                                              ABLOCK_VECTOR_DIM,
                                                              ABLOCK_SCALAR_VEC,
                                                              ABLOCK_SCALAR_VEC_K1,
                                                              ABLOCK_LDS_EXTRAM,
                                                              BBLOCK_CLUSTER_LENS,
                                                              BBLOCK_CLUSTER_ORDER,
                                                              BBLOCK_SRC_ORDER,
                                                              BBLOCK_VECTOR_DIM,
                                                              BBLOCK_SCALAR_VEC,
                                                              BBLOCK_SCALAR_VEC_AK1,
                                                              BBLOCK_LDS_EXTRAN,
                                                              CMPER_WAVE,
                                                              CNPER_WAVE,
                                                              CBLOCK_CLUSTER_LENS,
                                                              CNPER_BLOCK>;

    auto gemm = DeviceGemmInstance{};
    auto invoker = gemm.MakeInvoker();
    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    auto argument = gemm.MakeArgument(
        reinterpret_cast<const ADataType*>(a),
        reinterpret_cast<const BDataType*>(b),
        reinterpret_cast<CDataType*>(c),
        M,
        N,
        K,
        StrideA,
        StrideB,
        StrideC,
        a_element_op,
        b_element_op,
        c_element_op);

    if (!gemm.IsSupportedArgument(argument)) {
        printf("error shape = %ld %ld %ld TRANSA=%d TRANSB=%d \n",
               m, n, k, TRANSA, TRANSB);
        TORCH_CHECK(false, "device_gemm does not support this GEMM problem");
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    invoker.Run(argument, StreamConfig{stream, false});
}

__global__ void scale_bias_kernel(
    half* c,
    const float* scale,
    const half* bias,
    int64_t M,
    int64_t N,
    int64_t stride_cm,
    int64_t stride_cn,
    int has_scale,
    int has_bias)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = M * N;
    if (idx < total) {
        const int64_t m = idx / N;
        const int64_t n = idx - m * N;
        half val = c[m * stride_cm + n * stride_cn];
        if (has_scale) {
            val = __hmul(val, __float2half_rn(scale[0]));
        }
        if (has_bias) {
            val = __hadd(val, bias[n]);
        }
        c[m * stride_cm + n * stride_cn] = val;
    }
}

torch::Tensor scaled_mm_ck(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scale,
    const torch::Tensor& bias,
    bool has_scale,
    bool has_bias)
{
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be float16");
    TORCH_CHECK(b.scalar_type() == c10::ScalarType::Float8_e4m3fn, "b must be float8_e4m3fn");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "a and b shapes are incompatible");

    if (has_scale) {
        TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
        TORCH_CHECK(scale.numel() == 1, "scale must have one element");
        TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");
    }
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.numel() == b.size(1), "bias must have N elements");
        TORCH_CHECK(bias.scalar_type() == at::kHalf, "bias must be float16");
    }

    auto b_half = torch::empty({b.size(0), b.size(1)}, b.options().dtype(at::kHalf));
    auto c = torch::empty({a.size(0), b.size(1)}, a.options().dtype(at::kHalf));

    const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(b.data_ptr());
    half* b_half_ptr = reinterpret_cast<half*>(b_half.data_ptr<at::Half>());
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int kThreads = 256;
    const int64_t b_count = b.numel();
    const int64_t blocks = (b_count + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(
        fp8_to_half_kernel,
        dim3(static_cast<uint32_t>(blocks), 1, 1),
        dim3(kThreads, 1, 1),
        0,
        stream.stream(),
        b_ptr,
        b_half_ptr,
        b_count);

    const bool use_padding = ((a.size(0) % 256 != 0) || (b.size(1) % 128 != 0) || (a.size(1) % 64 != 0));
    const auto a_ptr = reinterpret_cast<const at::Half*>(a.data_ptr<at::Half>());
    const auto b_gemm_ptr = reinterpret_cast<const at::Half*>(b_half.data_ptr<at::Half>());
    auto c_ptr = reinterpret_cast<at::Half*>(c.data_ptr<at::Half>());
    constexpr auto transa = 'n';
    constexpr auto transb = 'n';
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (use_padding) {
        gemm_impl_wmma_noswap<
            at::Half,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1, 8>,
            8,
            true,
            false,
            false>(transa, transb, a.size(0), b.size(1), a.size(1), alpha, a_ptr,
                   a.stride(0), b_gemm_ptr, b_half.stride(0), beta, c_ptr, c.stride(0));
    } else {
        gemm_impl_wmma_noswap<
            at::Half,
            256,
            128,
            256,
            64,
            8,
            16,
            16,
            4,
            4,
            S<4, 64, 1>,
            S<1, 0, 2>,
            S<1, 0, 2>,
            2,
            8,
            8,
            true,
            S<4, 64, 1>,
            S<0, 2, 1>,
            S<0, 2, 1>,
            1,
            1,
            8,
            true,
            1,
            1,
            S<1, 32, 1, 8>,
            8,
            false,
            false,
            false>(transa, transb, a.size(0), b.size(1), a.size(1), alpha, a_ptr,
                   a.stride(0), b_gemm_ptr, b_half.stride(0), beta, c_ptr, c.stride(0));
    }

    if (has_scale || has_bias) {
        const int64_t total = a.size(0) * b.size(1);
        const int64_t sb_blocks = (total + kThreads - 1) / kThreads;
        hipLaunchKernelGGL(
            scale_bias_kernel,
            dim3(static_cast<uint32_t>(sb_blocks), 1, 1),
            dim3(kThreads, 1, 1),
            0,
            stream.stream(),
            reinterpret_cast<half*>(c_ptr),
            has_scale ? scale.data_ptr<float>() : nullptr,
            has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr,
            a.size(0),
            b.size(1),
            c.stride(0),
            c.stride(1),
            has_scale ? 1 : 0,
            has_bias ? 1 : 0);
    }

    return c;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("scaled_mm_ck", &scaled_mm_ck, "Scaled mixed-precision matmul (CK)");
}
