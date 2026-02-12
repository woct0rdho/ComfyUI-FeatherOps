# CK Reference (This Repo)

This file is the CK-specific reference for `kernel/ck/ck_kernel.cu` and related CK source.

## Scope

- Wrapper source: `kernel/ck/ck_kernel.cu`
- Python loader: `kernel/ck/ck_kernel.py`
- CK source tree: `~/rocm-libraries/projects/composablekernel`

## End-to-End Flow in `ck_kernel.cu`

1. `fp8_to_half_kernel`: converts full `B` from fp8(e4m3fn) to fp16 in global memory.
2. CK GEMM: `gemm_impl_wmma_noswap<...>()` instantiates `DeviceGemmWmma_CShuffle<...>`.
3. Optional epilogue: `scale_bias_kernel`.

Important: CK GEMM here receives fp16 `B` (already upcast), not in-kernel fp8->fp16 fused into GEMM.

## Concrete CK GEMM Template Instance

From `kernel/ck/ck_kernel.cu` (both padded and non-padded paths share the same core tuning):

- `BlockSize=256`
- `MPerBlock=128`
- `NPerBlock=256`
- `KPerBlock=64`
- `K1=8`
- `MPerWmma=16`, `NPerWmma=16`
- `MRepeat=4`, `NRepeat=4`
- A block transfer:
  - `ABlockClusterLengths = Sequence<4,64,1>`
  - `ABlockClusterOrder = Sequence<1,0,2>`
  - `ABlockSrcAccessOrder = Sequence<1,0,2>`
  - `ABlockTransferSrcVectorDim = 2`
  - `ABlockTransferSrcScalarPerVector = 8`
  - `ABlockTransferDstScalarPerVector_K1 = 8`
  - `ABlockLdsAddExtraM = true`
- B block transfer:
  - `BBlockClusterLengths = Sequence<4,64,1>`
  - `BBlockClusterOrder = Sequence<0,2,1>`
  - `BBlockSrcAccessOrder = Sequence<0,2,1>`
  - `BBlockTransferSrcVectorDim = 1`
  - `BBlockTransferSrcScalarPerVector = 1`
  - `BBlockTransferDstScalarPerVector_K1 = 8`
  - `BBlockLdsAddExtraN = true`
- C-shuffle:
  - `CShuffleMRepeatPerShuffle=1`
  - `CShuffleNRepeatPerShuffle=1`
  - `CShuffleBlockTransferClusterLengths = Sequence<1,32,1,8>`
  - `CShuffleBlockTransferScalarPerVector_NPerBlock=8`

## Instantiation Path to Device/Grid Code

- `DeviceGemmWmma_CShuffle` aliases `GridwiseGemm_Wmma` in:
  - `~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp`
- Grid-level A/B copy and WMMA pipeline are in:
  - `~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma.hpp`
- Thread-group transfer primitive is:
  - `~/rocm-libraries/projects/composablekernel/include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp`

## How A Loading Is Optimized (Current CK Instance)

### 1) LDS path is enabled for A

In `device_gemm_wmma.hpp`, `AEnableLds_auto` is disabled only for a narrow `NWaves==1` case.
With this instance:
- `NWaves = NPerBlock / (NRepeat * NPerWmma) = 256 / (4*16) = 4`
- so `AEnableLds = true`.

### 2) A LDS descriptor uses extra-M padding

In `gridwise_gemm_wmma.hpp`, `MakeABlockDescriptor()` for `AEnableLds` + `ABlockLdsExtraM=true` uses:
- shape: `(K0PerBlock, MPerBlock, K1) = (8, 128, 8)`
- stride: `((MPerBlock+1)*K1, K1, 1) = (1032, 8, 1)`

This `M+1` padding is the bank-conflict mitigation used by this CK path.

### 3) A global->LDS transfer is compile-time vectorized

`a_blockwise_copy` instantiates:
- `ThreadGroupTensorSliceTransfer_v4r1<...>`
- block slice lengths: `Sequence<K0PerBlock, MPerBlock, K1> = Sequence<8,128,8>`
- thread cluster lengths: `Sequence<4,64,1>` (256 threads total)
- source access/vector controls:
  - `SrcVectorDim = 2`
  - `SrcScalarPerVector = 8`
- destination vector controls:
  - `DstVectorDim = 2`
  - `DstScalarPerVector = 8`

So A-copy is designed as vectorized read and vectorized LDS write, with thread-cluster mapping fixed at compile time.

### 4) Legality checks enforce vectorization assumptions

`IsSupportedArgument()` in `device_gemm_wmma.hpp` checks:
- For row-major A with `SrcVectorDim=2`, requires `KRaw % ABlockTransferSrcScalarPerVector == 0`.
- Here that means `KRaw % 8 == 0`.

## What We Verified at Symbol/Artifact Level

From `kernel/ck/build/scaled_mm_ck_ext/ck_kernel.cuda.o`:
- Symbols show instantiated `DeviceGemmWmma_CShuffle<...>` with exactly the parameters above.
- Both `GemmSpecialization::Default` and `GemmSpecialization::MNKPadding` specializations are present.
- `kernel_gemm_wmma< GridwiseGemm_Wmma<...> >` device stubs are present.
- `.hip_fatbin` section exists (device code is embedded).

This is enough to prove the C++ template instantiation path and selected CK tuning.

## About HIP/ASM-Level Understanding

Current status:
- We have exact template-to-gridwise mapping and can explain A-load strategy precisely at CK source level.
- We have symbol-level proof that the tuned instance is compiled into the extension object.

Not yet extracted here:
- Final gfx1151 ISA listing (`global_load*`, `ds_write*`, `ds_read*`, `v_wmma*`) from the embedded code object.

If needed later, next step is to extract/disassemble the AMDGPU code object from `.hip_fatbin` and annotate the A-load instructions directly.

## Direct Implications for HIP Kernel Work

- CK’s A-path uses three coordinated pieces together:
  1. vectorized global read,
  2. vectorized LDS write,
  3. `M+1` LDS padding.
- Matching only one of these in HIP is usually insufficient; parity work should treat them as one bundle.
