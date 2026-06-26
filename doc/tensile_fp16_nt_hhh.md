# TensileLite FP16 NT HHH Plan

## Current State

- Goal: optimize NT HHH (fp16@fp16, fp16 accumulator) GEMM in hipBLASLt TensileLite on gfx1151 with input size `8192^3`.
- Baseline: `rocm_wmma_gemm` is `41.4 TFLOP/s` median and `41.7 TFLOP/s` mean.
- Current best TensileLite HHH candidate: `hhh_tlds2_mt128x128_pad128x8`, valid, two-run hot-loop combined median `42.133 TFLOP/s` and mean `41.944 TFLOP/s`.
- Main remaining work: profile and inspect the new TLDS2 padded hot-loop winner before adding a gfx1151 NT HHH logic entry.

## Operating Notes

- Use `~/rocm_wmma_gemm/build/` as the `rocm_wmma_gemm` build directory.
- Keep temporary TensileLite fp16 NT files under `~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/`.
- Invoke the runner with `bash`. `tmp_tensile_fp16_nt_hhh/scripts/run_fp16_nt_variant.sh` is not executable.
- Use `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/llvm/bin/llvm-objdump` for code-object disassembly. This build does not accept `--show-raw-insn`.
- When in doubt on WMMA/register semantics, read `~/rdna35-isa-markdown/`.
- After meaningful tuning steps, update this file before continuing.

## Layout Mapping

Public benchmark `NT` maps to TensileLite as follows:
- A is column-major physical storage, so hipBLASLt opA is `HIPBLAS_OP_N`.
- B is row-major physical storage, so hipBLASLt opB is `HIPBLAS_OP_T`.
- TensileLite operation is `Cijk_Ailk_Bjlk`.
- TensileLite problem type uses `TransposeA: False`, `TransposeB: True`.

Useful references:
- FeatherOps layout detection: `kernel/hip/hipblaslt_kernel_fp16.cu:124-151`.
- hipBLASLt op forwarding to TensileLite: `~/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/tensile_host.cpp:726-729`.
- TensileLite GEMM index assignment: `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Problem.py:1046-1063`.
- Existing gfx1151 NT logic: `~/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/GridBased/gfx1151_Cijk_Ailk_Bjlk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml`.

## Baseline Facts

### rocm_wmma_gemm NT

- Selected config source: `~/rocm_wmma_gemm/rocm_wmma_gemm/config/gemm_config_gfx1151.json`.
- Instantiation: `~/rocm_wmma_gemm/build/rocm_wmma_gemm/src/kernel_inst/kernel_inst_27.cpp`.
- Comparable symbol: `_ZN14rocm_wmma_gemm16kernel_gemm_implI6__halfS1_LNS_8m_layoutE0ELS2_1ELS2_0ELi4ELi2ELi4ELi4ELi16ELi128EE3runEPS1_PKS1_S6_iii`.
- Shape tuple: `{warps_m=4, warps_n=2, warp_tile_m=4, warp_tile_n=4, swizzle=16, bits=128}`.
- Block tile: `256x128x16`, `8` wave32 waves, `256` threads.
- WMMA builtin: `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32`, so the comparable target is HHH/half accumulation, not HHS/fp32 accumulation.
- Profiled resources: `192` VGPR, `128` SGPR, `24576` LDS bytes, `workgroup_x=256`.
- Profiled LDS conflict: near zero, `LDSBankConflict=0.177778` in the PMC run.
- Resource DB: `tmp_tensile_fp16_nt_hhh/logs/rocm_wmma_nt_resources/rocm_wmma_nt_resources_results.db`.
- PMC DB: `tmp_tensile_fp16_nt_hhh/logs/rocm_wmma_nt_pmc/rocm_wmma_nt_pmc_results.db`.
- Full disassembly from `llvm-objdump -d --demangle`: `~/.local/share/opencode/tool-output/tool_ea556e1840015VC0v2SIcTFSeF`.

### rocm_wmma LDS Behavior

- A is staged col-major as a `256x16` tile. B is staged row-major as a `16x128` tile.
- Input staging uses vector LDS stores such as `ds_store_b128`.
- WMMA fragment reads use scalar/sub-dword LDS reads such as `ds_load_u16_d16` and `ds_load_u16_d16_hi`.
- A low/high halves are separated by `512` bytes. B low/high halves by `256` bytes. `wm`/`wn` increments shift by `32` bytes.

## Current Best Retimed Candidate

### `hhh_tlds2_mt128x128_pad128x8`

- Status: validation passes in the generated client and in both fresh hot-loop retime passes.
- Key parameters: `MT128x128x16`, `TransposeLDS=2`, `NumElementsPerBatchStore=16`, `LdsBlockSizePerPadA/B=128`, `LdsPadA/B=8`, `PrefetchGlobalRead=2`, `PrefetchLocalRead=0`, `MIArchVgpr=1`.
- Default generated-config screen: median `28907.0 GFLOP/s`, mean `28876.7 GFLOP/s`.
- Hot-loop retime protocol: `num-benchmarks=10`, `num-warmups=20`, `num-enqueues-per-sync=10`, `num-syncs-per-benchmark=1`, `sleep-percent=0`, `hardware-monitor=False`.
- Hot-loop run 1: median `42221.1 GFLOP/s`, mean `41811.1 GFLOP/s`, min `39148.7`, max `42418.2`.
- Hot-loop run 2: median `42038.2 GFLOP/s`. Combined two-run median `42132.9 GFLOP/s`, mean `41943.7 GFLOP/s`, min `39148.7`, max `42418.2`, coefficient of variation `1.64%`.
- This candidate now reaches the `rocm_wmma_gemm` baseline regime under the same hot-loop-style repeated-enqueue timing, unlike the corrected TLDS0 `skipvp` candidate.
- Logs: `tmp_tensile_fp16_nt_hhh/logs/retime_hhh_tlds2_mt128x128_pad128x8_hot_loop.log`, `tmp_tensile_fp16_nt_hhh/logs/retime2_hhh_tlds2_mt128x128_pad128x8_hot_loop.log`.
- CSVs: `tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_hhh_tlds2_mt128x128_pad128x8_8192_hot_loop.csv`, `tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_hhh_tlds2_mt128x128_pad128x8_8192_hot_loop_rerun2.csv`.
- Client config: `tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_hhh_tlds2_mt128x128_pad128x8_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_H_UserArgs_00/00_Final/caches/717036c284d6/source/ClientParameters.ini`.

### Fresh Hot-Loop Rerun Shortlist

All rows below passed generated-client validation in both hot-loop retime passes. Values are CSV-parsed GFLOP/s from the per-solution performance column, not console `time-us`.

| Variant | Run 1 Median | Run 2 Median | Delta | Combined Median | Combined Mean | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `hhh_tlds2_mt128x128_pad128x8` | `42221.1` | `42038.2` | `-0.43%` | `42132.9` | `41943.7` | Best stable candidate |
| `hhh_tlds2_mt128x128_pad128x8_nepbs8` | `41144.3` | `40276.3` | `-2.11%` | `40507.7` | `40787.2` | Slower than NEPBS16 default |
| `hhh_tlds2_mt128x128_pad128x8_storesync1` | `40189.6` | `40288.8` | `+0.25%` | `40236.1` | `40409.5` | Stable, slower |
| `hhh_tlds2_mt128x128_pad128x8_nepbs32` | `39989.9` | `40149.6` | `+0.40%` | `40000.7` | `40034.9` | More variable |
| `hhh_tlds2_mt128x128_pad128x8_pgr1` | `38899.6` | `38791.2` | `-0.28%` | `38851.7` | `38894.6` | PGR1 loses |
| `hhh_tlds0_mt128x128_skipvp` | `32369.8` | `30528.8` | `-5.69%` | `31333.3` | `31443.8` | Valid but unstable/slower |
| `exact_hhh_tlds0_pgr0_skipvp` | `28697.8` | `29058.1` | `+1.26%` | `28891.2` | `28895.0` | Prior TLDS0 exact candidate |

## Prior Retimed TLDS0 Candidate

### `exact_hhh_tlds0_pgr0_skipvp`

- Status: validation passes under both the generated default config and hot-loop retimes.
- Prior documented samples `47330.8`, `47172.5`, `47450.9`, `47441.7`, `48001.1`, `47503.7`, `47568.9`, and `48101.4` were console `time-us` samples, not GFLOP/s.
- Reproduced original generated-config timing from the existing `ClientParameters.ini`: median `23118.0 GFLOP/s`, mean `23146.6 GFLOP/s`, derived median time `47561.0 us`.
- Original hot-loop retime with `num-warmups=20`, `num-enqueues-per-sync=10`, and `sleep-percent=0`: median `29278.3 GFLOP/s`, mean `29278.6 GFLOP/s`, derived median time `37553.7 us`.
- Fresh two-run hot-loop check: run medians `28697.8` and `29058.1 GFLOP/s`. Combined median `28891.2 GFLOP/s`, mean `28895.0 GFLOP/s`.
- Hot no-sleep single-enqueue control: median `29177.3 GFLOP/s`, mean `29133.2 GFLOP/s`, derived median time `37683.8 us`.
- Key parameters: `MT256x128x16`, `TransposeLDS=0`, `UnrollMajorLDSA/B=False`, `PrefetchGlobalRead=0`, `ScheduleGlobalRead=0`, `ScheduleLocalWrite=0`, `MIArchVgpr=1`.
- Codegen now derives A/B `ValuPack` allocation from the same local-read pack decision as `LocalRead.py`. No environment variable is required in `rocm-libraries` or the runner.
- Assembly metadata: `.amdhsa_next_free_vgpr 256`, `.amdhsa_group_segment_fixed_size 12288`.
- Local reads: direct `ds_load_u16` and `ds_load_u16_d16_hi` into `vgprValuA/B_X0_I0+*`.
- Sanity greps on generated assembly found no `ds_load_b128`, `v_perm_b32`, `vgprValu*_D*`, or `vgprValuPack` use in the local-read path.
- Log: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_exact_hhh_tlds0_pgr0_skipvp_8192.log`.
- Assembly: `tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_exact_hhh_tlds0_pgr0_skipvp_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_H_UserArgs_00/00_Final/caches/f92f30d7d093/source/build_tmp/SOURCE/assembly/Cijk_Ailk_Bjlk_H_UserArgs_MT256x128x16_MI16x16x1geogNEJ-QzTEBmGCouqDfKm0QneJhTRUMQApagtUvMM=.s`.
- Client config: `tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_exact_hhh_tlds0_pgr0_skipvp_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_H_UserArgs_00/00_Final/caches/f92f30d7d093/source/ClientParameters.ini`.

### Reproduced HHH Results

| Variant | Status | Median GFLOP/s | Mean GFLOP/s | LDS | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `hhh_tlds2_mt128x128_pad128x8` | pass, hot-loop retime x2 | `42132.9` | `41943.7` | `25600` | Current best. Default generated-config median was `28907.0 GFLOP/s` |
| `exact_hhh_tlds0_pgr0_skipvp` | pass, hot-loop retime x2 | `28891.2` | `28895.0` | `12288` | Prior TLDS0 exact candidate |
| `hhh_tlds2_mt128x128_pad128x8` | pass, default-config repro | `28935.6` | `28818.6` | `25600` | Reproduces the historical TLDS2 result |
| `hhh_tlds0_mt128x128_skipvp` | pass, default-config repro | `28280.1` | `28324.2` | `16384` | Prior candidate |
| `exact_hhh_tlds0_skipvp` | pass, default-config repro | `25183.3` | `25167.8` | `28672` | PGR2 exact tile |

Other artifact paths:
- `exact_hhh_tlds0_skipvp` log: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_exact_hhh_tlds0_skipvp_8192.log`.
- `exact_hhh_tlds0_skipvp` assembly: `tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_exact_hhh_tlds0_skipvp_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_H_UserArgs_00/00_Final/caches/68b33b7d3327/source/build_tmp/SOURCE/assembly/Cijk_Ailk_Bjlk_H_UserArgs_MT256x128x16_MI16x16x17ej4fNtPRUIShlPj4CHg6h8XnlLS6TC9znNaH0ocoEo=.s`.
- `hhh_tlds0_mt128x128_skipvp` log: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_hhh_tlds0_mt128x128_skipvp_8192.log`.
- `hhh_tlds0_mt128x128_skipvp` assembly: `tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_hhh_tlds0_mt128x128_skipvp_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_H_UserArgs_00/00_Final/caches/a7ddf71623b6/source/build_tmp/SOURCE/assembly/Cijk_Ailk_Bjlk_H_UserArgs_MT128x128x16_MI16x16x1kymoy7yfjCgrEwa2sNyCHUngJ_a7HZT0cbdVNS1ZeCI=.s`.

## Active Source Changes

Durable HHH correctness fixes:
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriterConversion.py`: fixes generated HHH helper source for sub-dword workspace loads. The original bug emitted invalid `float0 temp[NUM_GSU]` when `sizeof(half) / 4` rounded to zero.
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriter.py`: reserves a persistent `startVgprAlphaTmp` for half/no-HPA `MIArchVgpr` kernels.
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriterModules.py`: packs runtime scalar alpha for HHH `MIArchVgpr` before `v_pk_mul_f16`.

Structural performance/codegen change currently present:
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriter.py`: adds shared `localReadNeedsPack` and `localReadNeedsValuPack` helpers. A/B `ValuPack` allocation and pack scheduling now use the derived local-read pack decision instead of a gfx1151-specific skip predicate.
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py`: matching macro and tail-loop A/B `ValuPack` allocation use the shared derived helper.
- `tmp_tensile_fp16_nt_hhh/scripts/make_fp16_nt_speed_sweep.py`: named variants, including `skipvp` variants.
- `tmp_tensile_fp16_nt_hhh/scripts/run_fp16_nt_variant.sh`: stale `TENSILE_GFX1151_HHH_*` env-var plumbing was removed.

Important cleanup status:
- Temporary `DIAG HHH` markers and forced-output diagnostics should remain removed.
- Failed/research-only `rocm-libraries` experiments have been removed: TLDS2 scalar fragment-read `FRAGLR`, LDS base-shift, and exact compact LDS block-stride hooks.
- `rocm-libraries` and the local runner no longer depend on `TENSILE_GFX1151_HHH_*` environment variables for this path.

## HHH Correctness Root Cause

HHH initially generated and launched but produced zero/negative-zero output for `alpha=Two`. The decisive finding was that the post-GSU1 D-store path used `sgprAlpha` directly as a packed-half operand to `v_pk_mul_f16`.

Why that failed:
- Runtime alpha arrives as scalar f32 bits.
- `2.0f` is `0x40000000`.
- Treating those bits as packed f16 gives a zero low half, which zeroed HHH output lanes.

Durable fix:
- Copy `sgprAlpha` to a VGPR.
- Convert f32 alpha to f16 with `v_cvt_f16_f32`.
- Replicate it with `v_pack_b32_f16`.
- Use that packed-alpha VGPR as the `v_pk_mul_f16` operand.

Validation after fix:
- `exact_hhh_tlds2_k16`: passed all 8 validation runs.
- `exact_hhh_tlds2`: passed all 8 full-K validation runs.
- Expected fixed assembly sequence includes `v_mov_b32 v214, s[sgprAlpha]`, `v_cvt_f16_f32 v214, v214`, `v_pack_b32_f16 v214, v214, v214`, then `v_pk_mul_f16` using the packed alpha VGPR.

Ruled out during diagnosis:
- Missing launch, stale code object, D address/SRD/finalization, exec masks, simple low-vs-high half store, `SourceSwap`, long-loop accumulation, ordinary global/LDS input layout, hidden WMMA encoding mismatch, and raw accumulator mapping.

## TLDS0 `ValuPack` Bypass Reasoning

Pre-bypass TLDS0 was promising because `TransposeLDS=0` derives `UnrollMajorLDSA/B=False`, orienting LDS closer to `rocm_wmma_gemm` staging: A col-major `256x16`, B row-major `16x128`, and public tile dimensions contiguous in LDS.

The initial TLDS0 problem was VGPR pressure:
- `exact_hhh_tlds0`: assembly generation failed with `total vgpr: 280 not in [0, 256]`.
- `hhh_tlds0_mt128x128`: failed with `total vgpr: 284 not in [0, 256]`.
- `hhh_tlds0_mt128x128_pad128x8`: failed with the same `284` VGPR pressure.
- `PGR1`, `GRVWA4`, `GRVWB4`, `GRVW4`, and combined variants did not reduce the over-cap totals.

Inspection result:
- The generic non-unroll-major local-read path reserves a separate `ValuPack`/`Valu*_D*` footprint for half packing.
- Under the narrow gfx1151 HHH TLDS0 conditions, the actual local-read emission can directly fill final `ValuA/B` registers with `ds_load_u16` and `ds_load_u16_d16_hi`.
- Skipping the redundant `ValuPack` region lets TLDS0 fit at the gfx1151 `256` VGPR cap and preserves correctness in the tested variants.

Derived rule:
- A/B `ValuPack` is allocated only when A/B local reads need pack code for `bpe < 4`, non-unroll-major LDS, and non-LDSTr cases.
- The helper mirrors `LocalRead.py`: ECC-half or non-WMMA_V1 keeps the legacy sub-dword pack requirement. WMMA_V1 uses the selected local-read instruction width, so `blockWidth == 0.25` packs and the tested HHH `blockWidth == 0.5` path does not.
- Convert-after-DS, F32X emulation, and DTV pack/conversion paths still retain `ValuPack` allocation.
- MX-scale and metadata ValuPack paths remain unchanged because their allocation/macro code is separate.

## Historical Results To Keep

These results explain how the TLDS2 padded winner was found and which nearby YAML directions have already been weak.

| Variant or Group | Result | Why It Still Matters |
| --- | ---: | --- |
| `exact_hhh_tlds2` | `10554.8 GFLOP/s` | First full-K HHH-correct exact tile after alpha/helper fixes |
| `hhh_tlds2_mt128x128_nepbs16` | `13453.7 GFLOP/s` | Pre-padding best used for first resource/PMC comparison |
| `hhh_tlds2_mt128x128_pad128x8` | `28876.7 GFLOP/s` default, `42132.9 GFLOP/s` hot-loop combined median | Current best. Padding proved LDS conflict was significant and retiming exposed baseline-class sustained speed |
| `hhh_tlds2_mt128x128_pad128x8_fraglr` | `22365.1 GFLOP/s` | Naive scalar fragment-read replacement validated but regressed |
| `exact_hhh_tlds2_pad128x8` | `24214.9 GFLOP/s` | Exact `MT256x128` with padding improved but stayed below target |
| `exact_hhh_tlds2_compactlds` | `~10544 GFLOP/s` | Matched rocm-like `24576` LDS allocation but did not improve throughput |
| `exact_hhs_tlds2` | `~10.5 TFLOP/s` | HHS/fp32 accumulation is not the target arithmetic mode |
| `hhs_tlds2_mt128x128` | `~13.32 TFLOP/s` | HHS reference only. Register-heavy and not comparable to HHH baseline |

Focused TLDS2 negative conclusions:
- Symmetric `LdsBlockSizePerPadA/B=128` and `LdsPadA/B=8` was decisive, raising `MT128x128` HHH from about `13.45` to `28.88 TFLOP/s`.
- Nearby padding factors did not beat `pad128x8`: `pad256x8=27233.7`, `pad256x16=23306.0`, `pad128x16=22213.9`, `pad128x4=14369.5` GFLOP/s.
- One-sided padding helped but stayed lower: `padA128x8=20487.5`, `padB128x8=20353.6` GFLOP/s.
- Store batch, store priority/sync, PGR, SIA, LDS base shifts, and wave-tile/wave-group variants did not beat `pad128x8` in default screening. Fresh hot-loop retimes also showed `NEPBS8`, `NEPBS32`, `StoreSyncOpt=1`, and `PGR1` behind the default `NEPBS16/PGR2` winner.
- `PrefetchLocalRead=1` on the padded TLDS2 shape overflowed VGPRs with `total vgpr: 284 not in [0, 256]`.
- `PrefetchGlobalRead>=3` was rejected because it requires `DirectToLdsA/B` and `PrefetchLocalRead>=1`.

LDSTr/TLDS1 negative conclusion:
- `hhh_tlds1_ldstr_mt128x128` and `hhh_tlds1_ldstr_mt128x128_pad128x8` failed at solution enumeration.
- gfx1151 capability table reports no `HasLDSTr`, `HasLDSTrB128B16`, or `HasLDSTrB64B16`.
- Rejection included `TransposeLds requires TLUA=0 or TLUB=0`. This NT problem has both `TLUA=True` and `TLUB=True`.

Compact LDS negative conclusion:
- `exact_hhh_tlds2_compactlds` matched target LDS allocation in an earlier guarded experiment.
- It matched target LDS allocation: `LdsNumBytes=24576`, `LdsOffsetA_Blk=12288`, `LdsOffsetB_Blk=20480`, `StoreSwapAddr=true`.
- Assembly metadata was `.amdhsa_next_free_vgpr 256`, `.amdhsa_group_segment_fixed_size 24576`.
- Performance stayed around `10.54 TFLOP/s`, proving LDS resource/block-stride parity alone was not sufficient.

## Profile Data

Previously profiled TLDS2 candidates. These profiles predate the fresh hot-loop retime, so use them as layout/conflict diagnostics, not as current speed evidence:

| Kernel | Avg us | VGPR | SGPR | LDS | Workgroup | LDSBankConflict |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `hhh_tlds2_mt128x128_nepbs16` | `93787.7` | `224` | `128` | `16384` | `128x1x1` | `85.0` |
| `hhh_tlds2_mt128x128_pad128x8` | `38363.6` | `224` | `128` | `25600` | `128x1x1` | `43.75` overall, `50.0` nonzero-only |
| `rocm_wmma_gemm` NT baseline | `26709.3` | `192` | `128` | `24576` | `256x1x1` | `0.177778` |

Profile DBs:
- `tmp_tensile_fp16_nt_hhh/logs/hhh_mt128x128_nepbs16_resources/hhh_mt128x128_nepbs16_resources_results.db`.
- `tmp_tensile_fp16_nt_hhh/logs/hhh_mt128x128_nepbs16_pmc/hhh_mt128x128_nepbs16_pmc_results.db`.
- `tmp_tensile_fp16_nt_hhh/logs/hhh_mt128x128_pad128x8_resources/hhh_mt128x128_pad128x8_resources_results.db`.
- `tmp_tensile_fp16_nt_hhh/logs/hhh_mt128x128_pad128x8_pmc/hhh_mt128x128_pad128x8_pmc_results.db`.

Interpretation kept for future work:
- TLDS2 padding reduced LDS conflict materially. The old profile still showed much higher `LDSBankConflict` than `rocm_wmma_gemm`, but fresh hot-loop timing now reaches the baseline throughput regime, so the current winner needs a refreshed profile before drawing promotion or bottleneck conclusions.
- Naive TLDS2 scalar-fragment reads were correct but slower, so the remaining issue was not solved by replacing `ds_load_b128` with more scalar reads in the same layout.
- The retimed TLDS0 path changes both LDS orientation and local-read/register allocation, but it no longer beats the `rocm_wmma_gemm` baseline after correcting the CSV/console column mix-up.

## Run Protocol

Benchmark one or more named variants:

```bash
bash ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/scripts/run_fp16_nt_variant.sh <variant> [<variant> ...]
```

Retiming an already generated TensileLite candidate with hot-loop pacing:

```bash
~/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client \
  --config-file ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_<variant>_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_H_UserArgs_00/00_Final/caches/<cache>/source/ClientParameters.ini \
  --config-file ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/configs/<variant>_hot_loop_override.ini
```

The filenames above are examples of the folder layout. Prefer keeping this layout and only changing `<variant>`, `<cache>`, and the override `results-file` when retiming a new candidate.

Important current variants:
- `hhh_tlds2_mt128x128_pad128x8`: current best HHH candidate. Two-run hot-loop combined median `42.133 TFLOP/s`.
- `hhh_tlds2_mt128x128_pad128x8_nepbs8`, `hhh_tlds2_mt128x128_pad128x8_nepbs32`, `hhh_tlds2_mt128x128_pad128x8_storesync1`, and `hhh_tlds2_mt128x128_pad128x8_pgr1`: valid hot-loop-retimed comparisons that did not beat the current best.
- `exact_hhh_tlds0_pgr0_skipvp`: prior retimed TLDS0 candidate. Fresh two-run hot-loop combined median `28.891 TFLOP/s`.
- `exact_hhh_tlds0_skipvp`: valid PGR2 exact TLDS0 skipvp comparison.
- `hhh_tlds0_mt128x128_skipvp`: valid `MT128x128` TLDS0 skipvp comparison.
- `exact_hhh_tlds2_compactlds`: compact LDS negative control.

Generated configs intentionally include:
- `KeepBuildTmp: True`.
- `ForceRedoBenchmarkProblems: True`.
- `ForceRedoLibraryLogic: True`.
- `ForceRedoLibraryClient: True`.
- `NumBenchmarks: 8`.
- `NumWarmups: 3`.
- `NumElementsToValidate: 128`.
- `DataInitTypeAlpha: 2`, meaning `alpha=Two`.
- `DataInitTypeBeta: 1`, meaning `beta=Zero`.
- Generated `ClientParameters.ini` files use `num-enqueues-per-sync=1`, `num-warmups=3`, and `sleep-percent=300`, so they are default cool-loop timings, not hot-loop timings.

CSV parsing rule:
- Always use a Python script to read TensileLite CSV results. Do not visually copy values from the console or the first CSV column.
- The CSV header's first `GFlops` column is zero in these outputs. The actual GFLOP/s values are in the long per-solution kernel-name column after `TotalFlops`.
- The console table has both `time-us` and `gflops` columns. Several older TLDS0 `skipvp` doc entries accidentally copied `time-us` values as GFLOP/s.

Minimal CSV reader:

```python
import csv
from pathlib import Path
from statistics import mean, median

path = Path("<results.csv>")
with path.open(newline="") as f:
    rows = list(csv.reader(f))

header, data = rows[0], rows[1:]
total_idx = header.index(" TotalFlops") if " TotalFlops" in header else header.index("TotalFlops")
perf_cols = []
for idx, name in enumerate(header):
    if idx <= total_idx:
        continue
    try:
        values = [float(row[idx]) for row in data]
    except ValueError:
        continue
    if any(v != 0 for v in values):
        perf_cols.append((name.strip(), values))

name, gflops = perf_cols[-1]
print(name)
print("median_gflops", median(gflops))
print("mean_gflops", mean(gflops))
print("median_tflops", median(gflops) / 1000.0)
```

Caveat:
- TensileLite can still print `buildSourceCodeObjectFile ... [cache hit]`. Trust the loaded `.co` and disassembly, not regenerated `.s` alone, when validating codegen changes.

Direct same-input benchmark runner:

```bash
~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++ \
  -x hip --offload-arch=gfx1151 \
  --rocm-path=~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel \
  -I~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/include \
  -L~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/lib \
  -D__HIP_PLATFORM_AMD__ -DNO_PYTORCH -O3 -std=c++17 \
  ~/ComfyUI-FeatherOps/kernel/hip/hip_kernel_fp16.cu \
  ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/scripts/same_input_hip_tensile_hhh_nt.cu \
  -lamdhip64 \
  -o ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/scripts/same_input_hip_tensile_hhh_nt
```

Run a small same-input validation first:

```bash
~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/scripts/same_input_hip_tensile_hhh_nt \
  --m 256 --n 256 --k 256 \
  --validate-elems 65536 --tolerance 512 --hip-ext-launch
```

Run the current best direct benchmark with the same hot-loop shape as the TensileLite client retime:

```bash
~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/scripts/same_input_hip_tensile_hhh_nt \
  --m 8192 --n 8192 --k 8192 \
  --bench --warmup 20 --iters 10 --enqueues-per-sync 10 \
  --tensile-first --hip-ext-launch
```

Direct runner notes:
- The default code object and kernel name are the current best `hhh_tlds2_mt128x128_pad128x8` artifacts under cache `717036c284d6`.
- The runner launches the generated `.co` directly and compares against `kernel/hip/hip_kernel_fp16.cu` for validation. It does not use a CPU reference.
- The runner defaults to TensileLite-style `Half` random inputs: integer values in `[-3, 3]`, matching generated-client `init-a=Random` and `init-b=Random`. Use `--small-random-inputs` only for diagnostic comparisons against the older direct-runner distribution `[-0.25, 0.25]`. That distribution measured about `39.4 TFLOP/s` and does not reproduce the TensileLite client result.
- Runtime metadata must match the generated solution: `StaggerU=32`, `staggerStrideShift=3`, `GSU=1`, `WorkGroupMapping=16`, `WorkGroupMappingXCC=1`, and `cu_count` in `internalArgs1`.
- The direct kernarg buffer includes the promoted f32 `alpha` at offset `96` and the TensileLite `alpha_2` half payload at offsets `100..101`.
- Matching run reproduced the client-speed regime: median `42.390 TFLOP/s`, mean `42.342 TFLOP/s`, range `41.955 - 42.768 TFLOP/s`.
- Optional pointer-coloring controls are available for debugging: `--offset-a`, `--offset-b`, `--offset-c`, and `--offset-d` take byte offsets applied to the active device pointers.

Input-value speed sensitivity:
- The same direct kernel, launch metadata, visible kernargs, HIP runtime, and low address bits measured about `39.4 TFLOP/s` with the older `[-0.25, 0.25]` uniform-float direct inputs, but `42.390 TFLOP/s` median with TensileLite-style half random inputs in `[-3, 3]`.
- This is an input-distribution effect, not a NaN failure. Full-output validation over all `67,108,864` outputs found `mismatches(abs>tolerance)=0` for both modes, and the validator counts NaN `abs_diff` as a mismatch.
- Full-output `[-3, 3]` validation: `bit mismatches=70479`, `max abs diff=48.0`, no NaNs observed.
- Full-output `[-0.25, 0.25]` validation: `bit mismatches=65337681`, `max abs diff=0.53125`, no NaNs observed.
- This is explained by transistor switching power, see https://www.thonking.ai/p/strangely-matrix-multiplications
- Do not compare direct speed against TensileLite-client speed unless the direct input distribution matches the generated-client initializer. For diagnostic sweeps, record the input mode alongside every timing result.

## Important Artifacts

Core scripts:
- Variant generator: `tmp_tensile_fp16_nt_hhh/scripts/make_fp16_nt_speed_sweep.py`.
- Variant runner: `tmp_tensile_fp16_nt_hhh/scripts/run_fp16_nt_variant.sh`.
- TensileLite client: `~/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client`.
- Hot-loop override example: `tmp_tensile_fp16_nt_hhh/configs/<variant>_hot_loop_override.ini`.
- Hot-loop rerun override example: `tmp_tensile_fp16_nt_hhh/configs/<variant>_hot_loop_rerun2_override.ini`.
- No-sleep single-enqueue and cooldown override examples can follow the same `tmp_tensile_fp16_nt_hhh/configs/<variant>_<mode>_override.ini` pattern.

Key logs:
- Current best generated-client log example: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_<variant>_8192.log`.
- Current best hot-loop retime log examples: `tmp_tensile_fp16_nt_hhh/logs/retime_<variant>_hot_loop.log`, `tmp_tensile_fp16_nt_hhh/logs/retime2_<variant>_hot_loop.log`.
- Current best hot-loop CSV examples: `tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_<variant>_8192_hot_loop.csv`, `tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_<variant>_8192_hot_loop_rerun2.csv`.
- Valid TLDS0 PGR2 exact: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_exact_hhh_tlds0_skipvp_8192.log`.
- Valid TLDS0 `MT128x128`: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_hhh_tlds0_mt128x128_skipvp_8192.log`.
- Best TLDS2 YAML generation baseline: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_hhh_tlds2_mt128x128_pad128x8_8192.log`.
- TLDS2 scalar-fragment negative control: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_hhh_tlds2_mt128x128_pad128x8_fraglr_8192.log`.
- Compact LDS negative control: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_exact_hhh_tlds2_compactlds_8192.log`.
- Fixed HHH full-K base: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_exact_hhh_tlds2_8192.log`.
- Fixed HHH K=16 base: `tmp_tensile_fp16_nt_hhh/logs/hhs_nt_exact_hhh_tlds2_k16_8192.log`.

Source files touched or repeatedly inspected:
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriterConversion.py`.
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriter.py`.
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriterModules.py`.
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/KernelWriterAssembly.py`.
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/Components/LocalRead.py`.
- `~/rocm-libraries/projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Solution.py`.
- `~/rocm_wmma_gemm/rocm_wmma_gemm/include/rocm_wmma_gemm/kernel/kernel.hpp`.
- `~/rocm_wmma_gemm/rocm_wmma_gemm/include/rocm_wmma_gemm/kernel/load.hpp`.
- `~/rocm_wmma_gemm/rocm_wmma_gemm/include/rocm_wmma_gemm/kernel/fragment.hpp`.

## Next Steps

- Re-profile `hhh_tlds2_mt128x128_pad128x8` under the same hot-loop pacing used for the two-run result. Capture resources and PMC counters, especially `LDSBankConflict`, `VALUInsts`, L2 hit behavior, occupancy, and kernel duration. Compare against the existing `rocm_wmma_gemm` profile and, if needed, refresh the `rocm_wmma_gemm` hot-loop baseline in the same session.
- Disassemble the loaded `.co` for the current winner, not just regenerated `.s`. Compare the steady loop against `rocm_wmma_gemm`: local-read instruction mix, LDS address strides, WMMA issue cadence, wait/barrier placement, store path, VGPR/SGPR layout, and LDS allocation.
- Run a focused YAML sweep around the current winner rather than a broad search. Candidate axes: `StaggerU/StaggerUStride`, `WorkGroupMapping`, `StorePriorityOpt`, `StoreSyncOpt`, `NumElementsPerBatchStore`, `PrefetchGlobalRead` including a `PGR0` probe if accepted, and small nearby LDS pad/block adjustments. Retain only variants that pass validation and beat the current winner under two hot-loop retimes.
- Test resource/scheduling variants only if they are likely to address a measured bottleneck: `1LDSBuffer`, `NumLdsBlk`, `MIArchVgpr=0`, `DepthU=32`, and limited `MT256x128`/`MT128x256` retimes. Do not revive wide rectangular or depth-64 sweeps unless profile data shows the `MT128x128x16` winner is limited by occupancy or loop overhead.
- If profiling shows latency bubbles that `PLR1` could hide, investigate why `PrefetchLocalRead=1` overflows at `284` VGPR for the padded TLDS2 shape. Treat this as a codegen/register-lifetime task, not just a YAML sweep.
- Keep avoiding NT `VectorWidthB>=2` winner claims until the documented wrong-lane mapping class is fixed and revalidated. If fixed, retest rocBLAS-like `VWB2`/PLR scheduling as a separate candidate family.
- Before adding a tuned gfx1151 NT HHH logic entry, validate the chosen candidate through the intended hipBLASLt/FeatherOps path, reconfirm CSV-parsed GFLOP/s, and archive the exact code object, YAML, client config, and hot-loop CSVs.

Useful profile command template:

```bash
rocprofv3 --kernel-trace --stats \
  --pmc L2CacheHit VALUInsts LDSBankConflict \
  -d ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/logs/<out_dir> \
  -o <prefix> -- \
  ~/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client \
  --config-file ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/outputs/hhs_nt_<variant>_8192_out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_H_UserArgs_00/00_Final/caches/<cache>/source/ClientParameters.ini \
  --config-file ~/ComfyUI-FeatherOps/tmp_tensile_fp16_nt_hhh/configs/<variant>_hot_loop_override.ini
```

The profile command uses example filenames. Keep the existing output/config folder layout and substitute the variant/cache names for the candidate under test.

## Patch Policy

- Keep generator/codegen patches evidence-driven and minimal.
- Do not preserve temporary diagnostics after their result is recorded.
- Avoid environment-variable-controlled codegen for final logic generation. Use structural predicates or explicit solution parameters.
- Do not reject a faster candidate solely because it exceeds the `rocm_wmma_gemm` resource envelope, but use resource/counter data to explain regressions and risks.
