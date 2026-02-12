# torch.compile tuned-kernel findings on gfx1151 (N=8192)

## AI quick facts (read this first)

- Graph path at `N=8192` is 3 kernels, not 1 fused kernel:
  1. Triton fp8->fp16 cast (`triton_poi_fused__to_copy_0`)
  2. GEMM (`extern_kernels.mm` -> ATen -> rocBLAS/Tensile asm kernel)
  3. Triton fused epilogue (`mul(scale)+add(bias)`)
- CK backend is not used for this run on gfx1151; GEMM is rocBLAS/Tensile, not inductor CK.
- Selected GEMM symbol:
  - `Cijk_Ailk_Bljk_HHS_BH_MT128x128x32_MI16x16x16x1_SN_1LDSB1_..._PGR1_PLR0_SIA3_SU32_SUS256_..._WSGRA1_WSGRB1_WS32_WG32_4_1_...`
- Kernel launch/resources (steady run):
  - `grid=(8192,64,1)`, `workgroup=(128,1,1)`, `lds=17408 B`, `vgpr=256`, `sgpr=128`
  - GEMM avg time: `40.186 ms`
- End-to-end steady timing (3 kernels):
  - fp8 cast: `0.952 ms`
  - GEMM: `40.186 ms`
  - epilogue: `1.039 ms`
  - total: `42.176 ms`
- Pipeline/scheduling knobs encoded in the symbol and why they matter:
  - `PGR1`: global-read prefetch enabled
  - `PLR0`: no extra local-read prefetch depth
  - `SIA3`: MFMA-level scheduling/interleaving policy
  - `1LDSB1`: single-LDS-buffer pipeline mode
  - `SU32` + `SUS256`: stagger-U offseting for global-read start
  - `WSGRA1` + `WSGRB1`: wave-separated global-read assignment
- Low-level implementation pointers in ROCm Tensile source:
  - parameter semantics: `~/rocm-libraries/shared/tensile/Tensile/Common.py`
  - SIA3 scheduling and GR/LW per-MFMA shaping: `~/rocm-libraries/shared/tensile/Tensile/KernelWriter.py`
  - stagger-U SRD offset/wrap logic: `~/rocm-libraries/shared/tensile/Tensile/KernelWriterAssembly.py`
- Main optimization implication for our HIP kernel:
  - matching tile/workgroup alone is insufficient; the remaining gap is mostly from fine-grained pipeline scheduling/dataflow, not occupancy.

## Scope and artifacts

This analysis focuses on the **steady-state tuned kernel** for `profile_scaled_mm_torch.py` at `N=8192`, on this machine (`gfx1151`).

Artifacts generated in this repo:

- `rocprof_torch_8192_steady/torch8192_steady_results.db`
- `rocprof_torch_8192_steady/run.log`
- `rocprof_torch_8192/torch8192_results.db` (earlier run with autotune activity)

Key source files inspected:

- `profile_scaled_mm_torch.py`
- `kernel/naive.py`
- `~/pytorch/torch/_inductor/kernel/mm.py`
- `~/pytorch/torch/_inductor/utils.py`
- `~/pytorch/torch/_inductor/config.py`
- `~/pytorch/torch/_inductor/codegen/rocm/ck_universal_gemm_template.py`
- `~/rocm-libraries/shared/tensile/docs/src/conceptual/kernel-parameters.rst`
- `~/rocm-libraries/shared/tensile/Tensile/SolutionStructs.py`

## What torch.compile generated for this graph

From generated TorchInductor code:

1. `triton_poi_fused__to_copy_0`: converts `b` from fp8 (`f8e4m3fn`) to fp16 into an intermediate `buf0`.
2. `extern_kernels.mm(arg0_1, buf0, out=buf1)`: GEMM.
3. `triton_poi_fused__to_copy_add_mul_1`: fused `mul(scale)` + `add(bias)` epilogue on `buf1`.

So for this graph, fp8-to-fp16 conversion is **not fused into GEMM**. It is a separate pointwise kernel before GEMM.

## Steady-state timing (rocprofv3 DB)

From `rocprof_torch_8192_steady/torch8192_steady_results.db`:

- `triton_poi_fused__to_copy_0`: 23 calls, avg `951.516 us`
- tuned GEMM kernel: 23 calls, avg `40185.627 us` (`40.186 ms`)
- `triton_poi_fused__to_copy_add_mul_1`: 23 calls, avg `1039.385 us`

Per-iteration steady total:

- `0.9515 ms + 40.1856 ms + 1.0394 ms = 42.1765 ms`

Throughput at `N=8192` (`2*N^3` FLOPs):

- GEMM-only: `27.361 TFLOPS`
- end-to-end for this 3-kernel pipeline: `26.069 TFLOPS`

## Tuned GEMM kernel selected in steady state

Kernel symbol:

`Cijk_Ailk_Bljk_HHS_BH_MT128x128x32_MI16x16x16x1_SN_1LDSB1_AMAS3_BL1_BS1_EPS1_GLVWA8_GLVWB8_GRVW8_GSU1_GSUASB_ISA1151_IU1_K1_KLA_LBSPPA0_LBSPPB128_LPA0_LPB8_LRVW16_MIAV1_MMFGLC_NLCA1_NLCB1_PGR1_PLR0_SIA3_SS1_SU32_SUS256_SVW4_TT4_64_TLDS1_UMLDSA0_UMLDSB1_USFGROn1_VAW2_VSn1_VW4_VWB2_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM8`

Launch/resources (from `kernels` table):

- `grid = (8192, 64, 1)`, `workgroup = (128, 1, 1)`
- effective workgroups: `(64, 64, 1)` = `4096`
- `lds_size = 17408 B`
- `vgpr_count = 256`, `sgpr_count = 128`
- min/avg/max duration: `37.657 ms / 40.186 ms / 41.316 ms`

Occupancy implication (using `GFX1151_REFERENCE.md` limits):

- Wave size is 32; `workgroup_x=128` => 4 waves/workgroup.
- VGPR-limited waves/SIMD: `floor(1536 / 256) = 6`.
- LDS-limited workgroups/CU: `floor(64KB / 17408) = 3` => 12 waves/CU.
- Both point to about `12/32 = 37.5%` CU wave occupancy.

This looks like a deliberate high-resource, high-throughput kernel choice (low occupancy but aggressive tile/pipeline).

## Decoding important kernel-name tokens

Mapping uses Tensile naming docs (`kernel-parameters.rst`) plus Tensile name/value abbreviation logic (`SolutionStructs.py`).

- `MT128x128x32`: macro-tile `128x128`, `DepthU=32`
- `MI16x16x16x1`: matrix instruction shape
- `WG32_4_1`: workgroup shape
- `TT4_64`: thread tile
- `WS32`: wavefront size 32
- `1LDSB1`: single LDS buffer enabled
- `BL1` / `BS1`: buffer load/store enabled
- `PGR1` / `PLR0`: prefetch global read enabled, local read prefetch setting 0
- `GRVW8` / `GLVWA8` / `GLVWB8`: global read vectorization
- `LRVW16`: local read vector width
- `LPA0` / `LPB8`: LDS padding A/B
- `LBSPPA0` / `LBSPPB128`: LDS block size per pad for A/B
- `SIA3`: schedule iteration algorithm 3
- `SU32` / `SUS256`: stagger-U parameters
- `TLDS1`: transpose LDS enabled
- `UMLDSA0` / `UMLDSB1`: unroll-major LDS layout for A/B
- `WSGRA1` / `WSGRB1`: wave-separate global read A/B enabled
- `VW4` / `VWB2` / `SVW4`: vector widths for compute/B/store
- `VAW2`: vector atomic width 2
- `USFGROn1`: `UseSgprForGRO = -1` (auto mode)
- `VSn1`: `VectorStore = -1` (auto mode)
- `GSU1` + `GSUASB`: global split-U = 1, algorithm `SingleBuffer`
- `KLA`: kernel language assembly
- `K1`: kernel flag in Tensile min naming
- `MIAV1`: matrix-instruction arch VGPR mode
- `MMFGLC`: `MemoryModifierFormat = GLC` (seen in rocBLAS Tensile logic YAMLs with this key/value)
- `ISA1151`: target ISA gfx1151

## CK template instantiation status for this run

`torch._inductor.kernel.mm.tuned_mm` can add ATen, Triton, CK, CKTile candidates. For CK GEMM path, `use_ck_gemm_template(...)` must pass.

For this environment, CK path is effectively disabled:

- default `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` is `ATEN,TRITON,CPP` (no CK backend selected)
- ROCm CK supported arch list is limited to `gfx90a/gfx942/gfx950` in current config, not `gfx1151`

Therefore the tuned path here is not an inductor CK template kernel. It is:

- `extern_kernels.mm` -> ATen `mm` -> rocBLAS/Tensile kernel (the `Cijk_...` symbol above).

## Low-level optimization observations relevant to your HIP kernel

1. torch.compile currently does a full-tensor fp8->fp16 conversion **before** GEMM.
2. That conversion kernel moves about `201 MB` per iteration (`N=8192`) and runs around `0.95 ms` (about `212 GB/s`), i.e. memory-bandwidth heavy.
3. There is a separate post-GEMM fused epilogue kernel (`~1.04 ms`).
4. GEMM itself is a heavily tuned wave32 Tensile kernel with large tile (`128x128x32`), high vectorization, and moderate occupancy (~37.5%).

Implication: your HIP design goal (load fp8 to shared/LDS and upcast inside GEMM rather than global pre-convert) directly targets a real overhead in torch.compile’s current path on gfx1151.

## New Findings (2026-02-10): delta vs current HIP kernel

Compared artifacts:

- torch steady: `rocprof_torch_8192_steady/torch8192_steady_results.db`
- HIP before launch-shape match: `rocprof_hip_8192/hip8192_results.db`
- HIP after launch-shape match: `rocprof_hip_8192_matched/hip8192_matched_results.db`

### Kernel timing/resource delta at N=8192

| Path | Avg time (ms) | VGPR | LDS (B) | Grid | Workgroup |
|------|---------------|------|---------|------|-----------|
| torch GEMM only (`Cijk_...`) | 40.186 | 256 | 17408 | (8192, 64) | (128, 1) |
| HIP fused kernel (before matching launch shape) | 40.272 | 192 | 16896 | (4096, 128) | (64, 2) |
| HIP fused kernel (after matching launch shape) | 41.082 | 192 | 16896 | (8192, 64) | (128, 1) |

Torch side non-GEMM kernels remain:

- fp8->fp16 pre-cast: `0.952 ms`
- post-GEMM fused epilogue: `1.039 ms`

So, launch-shape matching alone did not close the gap; it regressed HIP fused-kernel time in this run.

### Additional torch kernel metadata confirmed

From `kernel_symbols` in steady DB:

- selected GEMM symbol id/kernel id: `627`
- code object id: `8`
- `group_segment_size = 17408`, `arch_vgpr_count = 256`, `sgpr_count = 128`

Multiple nearby precompiled variants exist in the same code object (e.g. variants with `EPS0/EPS1`, `GLVWA4/8`, and `1LDSB0/1` in names), but runtime selected the `EPS1 + GLVWA8/GLVWB8 + 1LDSB1` variant above.

### Practical interpretation for HIP matching work

With `MT128x128x32` and `WG32_4_1` geometry now matched, the remaining differences are dominated by implementation details, especially:

1. fp8 load/convert scheduling inside the GEMM mainloop.
2. local read vectorization behavior (torch symbol indicates `LRVW16`).
3. wave-separated/staggered global read behavior (`WSGRA1`, `WSGRB1`, `SU32`, `SUS256`, `PGR1`).

These are the highest-priority targets if we want HIP to match/apply the same optimization class as torch’s selected Tensile kernel while preserving fp8-in-LDS conversion strategy.
