# TensileLite FP16 NT HHS Grid Tuning Plan

## Goal

Tune gfx1151 NT HHS non-AuxH GridBased kernels over a small representative Cartesian shape grid first, then expand toward the broad grid and workload extensions documented in `doc/input_shapes.md`. The `8192^3` result in `doc/tensile_fp16_nt_hhs.md` is a useful center point, but small, skinny, and very large-K matrices can have different performance characteristics.

## Workflow

- First generate an NT HHS non-AuxH tuning YAML over the 100-shape pilot grid below, with about 10 candidate configs per shape.
- After the pilot validates the candidate proposal, batching, winner extraction, and verification workflow, expand toward the 9,681-shape large-grid target union plus exact and K-banded workload additions from `doc/input_shapes.md`.
- Run TensileLite tuning with validation enabled for screening, then retime winners or high-risk shapes with the hot-loop protocol from `doc/tensile_fp16_nt_hhs.md`.
- Merge the resulting `3_LibraryLogic`, run the Tensile logic check, rebuild hipBLASLt, and verify correctness with generated `hipblaslt-bench -v` commands over exact tuned points plus off-grid GridBased-selection points.
- Compare old and new performance with the same generated `hipblaslt-bench` command grid, without `-v`, preferably through `clients/scripts/performance/hipblaslt-perf --run_sh` so CSVs include repeated-sample mean/median data.
- After NT HHS non-AuxH is tuned, copy the selected configs to NT HHS AuxH and regenerate/build as an AuxH problem type. Copy to NT BBS only as a bootstrap, and deploy BBS configs only after representative BF16 performance verification.

## Initial 100-Shape Pilot

Use this dense subset of decomposition block 5 from `tmp_tensile_fp16_nt_hhs/shape_data/large_grid_target_union_decomposition.json` before attempting the full 9,681-shape target:

```text
M:     [512, 640, 896, 1024]
N:     [128, 256, 512, 768, 1024]
batch: [1]
K:     [256, 512, 1024, 2048, 4096]
```

This gives `4 * 5 * 1 * 5 = 100` exact shapes. Every point is already in the large-grid target union, but the set avoids the very small, odd, highly skinny, and `8192`-special tails. Use it to tune and validate the candidate proposal machinery before scaling out.

## Tuning Time Estimate

The 100-shape pilot grid has total `sum(M*N*batch*K) = 65,531,805,696`, or `0.119` equivalent `8192^3` GEMMs. The 9,681-shape large-grid target union has total `sum(M*N*batch*K) = 16,355,524,682,555`, or `29.75` equivalent `8192^3` GEMMs. Using the current best `8192^3` hot-loop median from `doc/tensile_fp16_nt_hhs.md`, one `8192^3` GEMM is about `27.159 ms` at `40.484 TFLOP/s`.

For 10 candidate configs per pilot shape, the raw GEMM time is tiny compared with the harness overhead:

```text
single execution over all 100 pilot shapes for one candidate:
  0.119 * 27.159 ms = 3.23 ms

hot-loop protocol, 120 executions per candidate-shape:
  3.23 ms * 120 * 10 candidates = 3.88 s
```

The current one-candidate/one-shape generated-client path is overhead dominated. A cache-hit `8192^3` candidate takes about `6.0 s` wall time in the TensileLite runner, while the timed default GEMMs inside that run account for only about `0.32 s`. Treating the remaining `5.7-6.0 s` as per candidate-shape setup/runner overhead gives this pessimistic pilot estimate:

```text
100 pilot shapes * 10 candidates * 5.7-6.0 s
  = 95-100 min
  = 1.6-1.7 hours
```

Cold code-object builds or extra per-candidate validation can push the pilot toward roughly `2` hours in a per-candidate/per-shape process model. If the 100 shapes are batched by candidate and code objects are reused, the hard GEMM lower bound is only a few seconds; practical wall time should then be minutes to tens of minutes, driven by tensor initialization, validation, and runner bookkeeping.

For reference, the full 9,681-shape grid under the same 10-candidate assumption has this raw GEMM lower bound:

```text
single execution over all 9,681 shapes for one candidate:
  29.75 * 27.159 ms = 0.808 s

hot-loop protocol, 120 executions per candidate-shape:
  0.808 s * 120 * 10 candidates = 969.6 s = 16.2 min
```

The corresponding pessimistic per-candidate/per-shape full-grid estimate is:

```text
9,681 shapes * 10 candidates * 5.7-6.0 s
  = 153-161 hours
  = 6.4-6.7 days
```

Cold code-object builds or extra per-candidate validation can push the full grid toward roughly `8` days. Conversely, if shapes and candidate configs are batched into far fewer client invocations and code objects are reused, the hard GEMM lower bound is only about `16` minutes; practical wall time would then be driven by tensor initialization, validation, and runner bookkeeping rather than matmul. The grid tuning workflow should therefore avoid a per-candidate/per-shape process model where possible.

## Origami Prediction

Origami is ROCm's analytical GEMM configuration selector under `~/rocm-libraries/shared/origami`. It predicts and ranks GEMM configurations from problem size, data type, hardware properties, tile shape, matrix instruction, occupancy, vector widths, and a limited set of backend parameters. It is deterministic analytical modeling, not Bayesian optimization or evolutionary search.

There are two relevant paths:

- Generic Origami `rank_configs` supports gfx1151 in `shared/origami` and can rank a supplied candidate list.
- TensileLite's client-side `PredictionThreshold` path uses the `Formocast` simulator from Origami, but that simulator currently has hardware constants only for gfx942, gfx950, and gfx1201. On gfx1151 it should not be treated as usable until gfx1151 constants are added or the path is validated.

A local pilot for this NT HHS work is available at `tmp_tensile_fp16_nt_hhs/scripts/run_origami_nt_hhs_pilot.py`. It compiles `tmp_tensile_fp16_nt_hhs/scripts/origami_nt_hhs_ranker.cpp`, runs generic Origami predictions for selected NT HHS candidates and shapes, benchmarks the same candidates with TensileLite, and writes results under `tmp_tensile_fp16_nt_hhs/origami_pilot/`:

```bash
python3 tmp_tensile_fp16_nt_hhs/scripts/run_origami_nt_hhs_pilot.py --num-benchmarks 5
```

The pilot tested seven large-fork-space candidate configs, including a general-grid equivalent of the recent `8192^3` winner, across `8192^3`, `2048^3`, `8192x8192x512`, `16x4096x4096`, and `4096x16x4096`. Generic Origami did not predict any measured median winner in that set. For `8192^3`, it ranked the measured winner fifth of seven and ranked a `128x64 DU64` candidate first, even though that candidate measured much slower.

For now, do not use Origami as a hard pruning gate for this NT HHS grid tuning. It may still be useful as a weak feature, a candidate-ordering hint, or a diagnostic, but pruning to Origami's top-N candidates would discard known-good configs for the square and K-small cases in this task.

## Variant Policy

Tune the primary NT library as HHS without AuxH, using the same epilogue-capable problem type as the existing UserArgs large grids: `UseBias=1`, `UseScaleAlphaVec=1`, `Activation=True`, `ActivationType=hipblaslt_all`, fused activation, `UseScaleAB=""`, `UseScaleCD=False`, `Gradient=False`, `GroupedGemm=False`, and `Sparse=0`.

After tuning, copy the selected NT HHS configs to the matching NT HHS AuxH library and regenerate/build it as an AuxH problem type. `UseE`/AuxH matching is exact at inference time, so the AuxH and non-AuxH libraries both need entries even when they use the same selected configs.

It is acceptable to copy the NT HHS selected configs to NT BBS as a bootstrap, but do not deploy the BBS configs as final until representative BF16 performance has been verified. Existing gfx1151 data is mixed: TN HHS/BBS configs are effectively identical, while NN and current small NT HHS/BBS configs differ substantially.

## Fixed Problem Type

```yaml
OperationType: GEMM
DataType: H
DataTypeA: H
DataTypeB: H
DestDataType: H
ComputeDataType: S
HighPrecisionAccumulate: True
TransposeA: False
TransposeB: True
UseBeta: True
Batched: True
BiasSrc: D
UseBias: 1
BiasDataTypeList: [h]
UseE: False
UseScaleAlphaVec: 1
UseScaleAB: ""
UseScaleCD: False
Activation: True
ActivationType: hipblaslt_all
Gradient: False
GroupedGemm: False
Sparse: 0
SupportUserArgs: True
```

## Fork Vocabulary

This is a source vocabulary for candidate proposal, not a Cartesian search plan. The product of the listed axes is roughly `8.46M` configs per shape before rejection, or about `82B` configs over the 9,681-shape grid. That is impractical. We still need a proposal strategy that selects a small number of candidates, on the order of `10` per input shape, using the `8192^3` winner, existing TN/NN grid logic, shape buckets, and targeted probes for skinny or high-K cases.

```yaml
ForkParameters:
  - KernelLanguage: [Assembly]
  - WavefrontSize: [32]

  - MatrixInstruction:
    - [16, 16, 16, 1, 1, 1, 1, 2, 2] # MT32x32
    - [16, 16, 16, 1, 1, 1, 1, 1, 4] # MT16x64
    - [16, 16, 16, 1, 1, 1, 1, 4, 1] # MT64x16
    - [16, 16, 16, 1, 1, 2, 1, 2, 2] # MT64x32
    - [16, 16, 16, 1, 1, 1, 2, 2, 2] # MT32x64
    - [16, 16, 16, 1, 1, 2, 2, 2, 2] # MT64x64
    - [16, 16, 16, 1, 1, 2, 3, 2, 2] # MT64x96
    - [16, 16, 16, 1, 1, 3, 2, 2, 2] # MT96x64
    - [16, 16, 16, 1, 1, 3, 3, 2, 2] # MT96x96
    - [16, 16, 16, 1, 1, 4, 2, 2, 2] # MT128x64
    - [16, 16, 16, 1, 1, 2, 4, 2, 2] # MT64x128
    - [16, 16, 16, 1, 1, 4, 3, 2, 2] # MT128x96
    - [16, 16, 16, 1, 1, 3, 4, 2, 2] # MT96x128
    - [16, 16, 16, 1, 1, 4, 4, 2, 2] # MT128x128, current 8192^3 center
    - [16, 16, 16, 1, 1, 4, 4, 4, 1] # MT256x64
    - [16, 16, 16, 1, 1, 4, 4, 2, 4] # MT128x256
    - [16, 16, 16, 1, 1, 4, 4, 4, 2] # MT256x128

  - DepthU: [16, 32, 64]
  - GlobalSplitU: [1, 2, 4]
  - GlobalSplitUAlgorithm: [MultipleBuffer]

  - PrefetchGlobalRead: [1, 2, 0]
  - PrefetchLocalRead: [1, 0]
  - ScheduleGlobalRead: [1]
  - ScheduleLocalWrite: [1]
  - ScheduleIterAlg: [2, 3, 1]

  - WorkGroupMapping: [8, 5]
  - StaggerU: [32, 8, 0]
  - StaggerUStride: [256]
  - StaggerUMapping: [0, 1]

  - SourceSwap: [1, 0]
  - 1LDSBuffer: [1, 0]
  - ClusterLocalRead: [0, 1]
  - TransposeLDS: [0]

  - VectorWidthA: [1]
  - VectorWidthB: [2, 1]
  - GlobalReadVectorWidthA: [8, 4, 2, 1]
  - GlobalReadVectorWidthB: [8, 4, 2, 1]
  - LocalReadVectorWidth: [16]

  - StoreVectorWidth: [1]
  - StoreRemapVectorWidth: [0]
  - StorePriorityOpt: [True]
  - NumElementsPerBatchStore: [8]
  - StoreSyncOpt: [0]
  - GroupLoadStore: [False]

  - AssertFree0ElementMultiple: [1]
  - AssertFree1ElementMultiple: [1]
  - AssertSummationElementMultiple: [1]
```

Do not include these as fork axes: `InternalSupportParams`, `SupportUserArgs`, `Batched`, `UseBias`, `UseScaleAlphaVec`, `Activation`, `ActivationType`, `UseScaleAB`, `UseScaleCD`, `Gradient`, `GroupedGemm`, or `Sparse`.
