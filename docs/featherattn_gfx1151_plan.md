# FeatherAttn gfx1151 Forward Kernel

## Status

The gfx1151 forward kernel is production-qualified for dense non-causal FP16 attention with contiguous HND and NHD tensors and head dimensions 64 and 128. The accepted kernel body was qualified at commit `01454e3`; the later source cleanup changed file ownership and build organization but did not change forward arithmetic or dispatch.

The current result is:
- `168/168` public-contract cases pass.
- The authoritative 36-row performance matrix has `31/36` FeatherAttn wins over the FlashAttention AITER Triton kernel.
- Geometric-mean throughput is `32.258 TFLOPS` for FeatherAttn and `29.397 TFLOPS` for AITER, a `1.097x` Feather/AITER ratio.
- All 20 linked forward images stay at or below 191 used VGPRs, 32,768 bytes LDS, zero private memory, and zero SGPR/VGPR spills.
- No forward optimization experiment is currently open. New work must be justified by a measured bottleneck and an explicit promotion gate.

Hardware and execution assumptions:

| Item | Value |
| --- | --- |
| GPU | Radeon 8060S, gfx1151 |
| Compute units | 40 CUs, 20 WGPs |
| Wave size | 32 |
| Main comparison | FlashAttention AITER Triton `attn_fwd` |
| Arithmetic | FP16 WMMA with FP32 score and output accumulation |

## Public Contract

The public operation is:

```text
out = softmax(Q @ K^T / sqrt(D)) @ V
```

| Property | Current support |
| --- | --- |
| Input/output dtype | FP16 |
| Layouts | Contiguous HND `[B,H,N,D]` and NHD `[B,N,H,D]` |
| Head dimensions | 64 and 128 |
| Head count | Any positive count within checked launch/address limits |
| Query and KV lengths | Positive, independent, and not required to align to a tile |
| Attention | Dense, non-causal, equal query/KV heads |
| Unsupported | Masks, dropout, bias, ALiBi, windows, GQA/MQA, other dtypes, non-contiguous tensors |

The host validates dimensions, contiguity, layout, dtype, device, sequence/grid bounds, and signed-int32 addressability before narrowing any device-side index. Sequence length changes only grid size and the number of KV-loop iterations; it does not change per-workgroup LDS or the compiled kernel shape.

Accuracy is checked against AITER-backed FP32 reconstruction. The public elementwise gate is `atol=rtol=0.10` for `N_KV < 1024` and `atol=rtol=0.05` for `N_KV >= 1024`. The full test also requires finite output and bounded relative-L2 and normalized error.

## Current Design

### Tile and Ownership

| Parameter | Value |
| --- | ---: |
| Query tile `Br` | 128 rows |
| KV tile `Bc` | 64 rows |
| Workgroup | 256 threads |
| Waves per workgroup | 8 wave32 waves |
| Query ownership | One wave owns 16 query rows |
| Q storage in LDS | E5M2, 1 byte per value |
| K/V storage in LDS | FP16 |
| QK and persistent output state | FP32 |

One workgroup owns one 128-row query tile for one physical head. Each wave owns 16 query rows from score generation through softmax and output accumulation. This row ownership keeps the score fragments, probability conversion, and output fragments inside the producing wave and avoids a full score/probability LDS tile.

### Dataflow

- Load one 128-row Q tile, multiply by `log2(e) / sqrt(D)`, round to E5M2 with RNE, and store the compact values in lane-major LDS.
- Stage one 64-row K tile in FP16 LDS.
- Decode Q fragments to FP16 immediately before WMMA and compute transposed score fragments as `K * Q^T`. The transposed C layout gives each lane one query row for the local softmax reduction.
- Apply the KV-tail mask, perform the online softmax update in FP32, rescale the persistent FP32 output fragments, and convert the current probabilities directly from the WMMA C layout to WMMA A registers.
- Stage V with the rotating-shared swizzle and accumulate `P @ V` into the wave-owned FP32 output fragments.
- Convert the final output fragments to FP16 and store them with query-tail guards.

The accepted recurrence keeps a log-domain `lane_lse` state. A linear `(running_max, running_sum)` recurrence was correct and removed one logarithm, but it was slower on gfx1151 and is not used.

### LDS Layout

| D | Q8 LDS | Reused K/V LDS | Total LDS |
| ---: | ---: | ---: | ---: |
| 64 | 8,192 B | 8,192 B | 16,384 B |
| 128 | 16,384 B | 16,384 B | 32,768 B |

K and V share one lifetime-reused LDS region. K uses an XOR-swizzled row/chunk layout. For D64, V uses the Triton-style rotating phase

```text
n_chunk XOR ((d % 8) XOR ((d / 8) % 8))
```

which keeps D64 conflict-free at the profiler's resolution and limits D128 to a small repeatable residual conflict.

### Specialization and Dispatch

The kernel is compile-time specialized by:
- head dimension: D64 or D128;
- physical layout: HND or NHD;
- query tail: aligned or guarded;
- KV tail: aligned or guarded.

The base set contains 16 images. Four additional D64 NHD strided images implement the accepted long-sequence partition-aware mapping.

HND uses flattened batch/head ownership. NHD makes head the fastest grid axis so adjacent workgroups touch adjacent head offsets. Two bounded host policies handle long NHD tensors:
- D128 may split heads into sequential LLC-sized groups when the total K/V working set is large enough to benefit.
- D64 may use the strided physical-head mapping `group_index + local_head * group_count`. The selector avoids group counts divisible by eight because those strides recreate the gfx1151 memory-partition cliff.

All sublaunches run on the caller's current stream. Tail selection and grouping do not change the tensor layout or require a physical transpose.

### Q Decode Caching

D64 HND has enough register headroom to cache decoded Q fragments across KV tiles. Aligned, query-tail, and combined-tail images cache all four D16 fragments; the KV-tail image caches three. D64 NHD and all D128 images retain decode-on-use because their measured tradeoffs or resource pressure do not justify persistent decoded Q state.

### Source Organization

| File | Role |
| --- | --- |
| `kernel_attn/hip/hip_kernel.py` | Public wrapper and extension source list |
| `kernel_attn/hip/hip_kernel.cpp` | Shared forward/backward validation, dispatch, stream lookup, and Torch registration |
| `kernel_attn/hip/hip_kernel.h` | Raw forward/backward launch ABI |
| `kernel_attn/hip/featherattn_fwd_kernel.h` | Shared forward device template |
| `kernel_attn/hip/featherattn_fwd_aligned.cu` | Aligned instantiations |
| `kernel_attn/hip/featherattn_fwd_query_tail.cu` | Query-tail instantiations |
| `kernel_attn/hip/featherattn_fwd_key_tail.cu` | KV-tail instantiations |
| `kernel_attn/hip/featherattn_fwd_query_key_tail.cu` | Combined-tail instantiations |
| `kernel_attn/hip/featherattn_fwd_strided.cu` | Partition-aware D64 NHD instantiations |

## Resource Results

The table reports exact loaded gfx1151 metadata from the qualified production extension.

| Kernel group | Used VGPRs | LDS | Private/spills | Notes |
| --- | ---: | ---: | ---: | --- |
| D64 HND, all tail modes | 171-175 | 16,384 B | 0 / 0 | Expanded decoded-Q cache |
| D64 NHD, base and strided | 130-151 | 16,384 B | 0 / 0 | Includes four partition-aware images |
| D128 HND, all tail modes | 191 | 32,768 B | 0 / 0 | Resource-pinned |
| D128 NHD, all tail modes | 191 | 32,768 B | 0 / 0 | Resource-pinned; optional grouped launches |

The maximum SGPR count is 46. Base kernels use 56-byte arguments; strided D64 NHD kernels use 64-byte arguments. Every image uses wave32 and has zero private segment, scratch, and register spills.

## Current Benchmarks

### Method

The authoritative matrix uses batch one, heads `{16,32,56}`, sequence lengths `{4096,8192,16384}`, D `{64,128}`, and both physical layouts. Each provider uses `triton.testing.do_bench` with `warmup=25` and `rep=100`.

Forward throughput counts the two attention GEMMs:

```text
FLOPs = 4 * B * H * NQ * NKV * D
TFLOPS = FLOPs / elapsed_seconds / 1e12
```

`Feather / AITER` is the throughput ratio. Values above `1.000x` favor FeatherAttn. AITER consumes zero-copy transposed views for the HND comparison.

Raw matrix: `~/tmp/feather_attn/phase11_final/matrix/attn.csv`.

### Aggregate Results

| Layout | D | AITER geomean TFLOPS | Feather geomean TFLOPS | Feather / AITER | Feather wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 64 | 30.941 | 34.071 | 1.101x | 9/9 |
| HND | 128 | 31.354 | 34.078 | 1.087x | 8/9 |
| NHD | 64 | 29.248 | 30.643 | 1.048x | 6/9 |
| NHD | 128 | 26.322 | 30.434 | 1.156x | 8/9 |
| All | 64/128 | 29.397 | 32.258 | 1.097x | 31/36 |

### Representative Rows

These rows show the fast paths, the small regressions that remain, and the long-NHD grouping gains without reproducing the complete 36-row matrix.

| Layout | D | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HND | 64 | 16 | 4096 | 33.981 | 35.898 | 1.056x |
| HND | 64 | 56 | 16384 | 29.140 | 33.761 | 1.159x |
| HND | 128 | 16 | 4096 | 35.015 | 34.262 | 0.978x |
| HND | 128 | 56 | 4096 | 28.927 | 34.007 | 1.176x |
| NHD | 64 | 16 | 16384 | 31.943 | 31.197 | 0.977x |
| NHD | 64 | 32 | 16384 | 22.754 | 29.088 | 1.278x |
| NHD | 128 | 16 | 4096 | 31.842 | 31.202 | 0.980x |
| NHD | 128 | 32 | 8192 | 21.718 | 30.168 | 1.389x |
| NHD | 128 | 32 | 16384 | 21.566 | 27.393 | 1.270x |

Short, independent-tail, arbitrary-head, and batch-two cases are correctness requirements rather than part of this throughput matrix. The `168/168` public test covers lengths from 1 through 16,384, the `1023/1024/1025` tolerance boundary, independent query/KV tails, odd head counts, and batch two.

## Current Profile Results

### Accepted Optimization Effects

| Change | Measured effect | Current use |
| --- | --- | --- |
| Expanded D64 HND decoded-Q cache | `+0.789%` geomean, 9/9 wins; dynamic VALU `-3.39%`; LDS instructions `-3.83%` | Enabled for D64 HND |
| Partition-aware D64 NHD strided groups | `+21.280%` selected-domain geomean, 12/12 wins | Guarded long-NHD selector |
| D64 NHD H32/N16384 traffic | Fetch `15.830 -> 13.554 GiB`; GCEA reads `15.858 -> 13.460 GiB`; L2 hit `2.08% -> 12.93%` | Explains the `1.278x` AITER ratio at this row |
| Bounded D128 NHD LLC grouping | `1.135x` focused geomean over ungrouped, 9/9 wins; H32/N16384 fetch approximately `31.435 -> 14.822 GiB` | Enabled when the K/V working set crosses the LLC gate |

### LDS Bank Conflicts

| Layout/domain | AITER | Feather | Interpretation |
| --- | ---: | ---: | --- |
| D64 HND/NHD | 0.000% | 0.000% | Conflict-free at counter resolution |
| D128 H16/N4096 | 2.849% | 2.702% | Small repeatable residual |
| D128 NHD H32/N16384 | 2.855% | 2.702% | Grouped path does not add conflict |

The D128 residual is not the primary bottleneck: prior pressure profiles measured Feather `ALUStalledByLDS` at only `0.026-0.029%`. D128 is instead constrained by its 191-VGPR, 32-KiB image and by instruction/dependency work. Long NHD cases are primarily controller-traffic and cross-workgroup reuse problems, which is why launch grouping produced larger gains than local arithmetic edits.

## Accepted Work

| Work | Result |
| --- | --- |
| Row-owned 128x64x8 kernel | Accepted as the common D64/D128 topology |
| E5M2 Q in LDS with FP16 decode before WMMA | Accepted; reduces LDS without requiring unavailable FP8 WMMA |
| RNE Q encoding and log2 pre-scaling | Accepted numerical policy |
| Transposed `K * Q^T` score ownership | Accepted; removed the expensive full C-fragment transpose |
| Register online softmax and direct C-to-A probability conversion | Accepted |
| Rotating V LDS swizzle | Accepted; removes the severe store-conflict pattern |
| Compile-time query/KV tails | Accepted; aligned path remains unchanged |
| D64 specialization | Accepted at 16 KiB LDS |
| Native HND and NHD layouts | Accepted; no physical transpose |
| Head-fast NHD grid order | Accepted; removes the original partition aliasing |
| Bounded D128 NHD head grouping | Accepted for large K/V working sets |
| Expanded D64 HND decoded-Q cache | Accepted |
| Partition-aware D64 NHD strided grouping | Accepted |

## Rejected Work

| Work | Evidence | Decision |
| --- | --- | --- |
| Full score/probability LDS and scalar-row softmax | Poor ownership and occupancy; incompatible with the accepted row-owned design | Replaced structurally |
| In-thread V transpose with the original XOR layout | LDS conflict `29.4028`; `5.965 ms` versus AITER `4.402 ms` at H16/N4096/D128 | Rejected |
| Persistent linear `(m,l)` softmax state | Correct and spill-free, but `-0.938%` paired geomean | Rejected |
| Four-pair progressive QK LDS issue | HND `-2.263%`, NHD `-1.149%` | Rejected |
| Two-pair progressive QK LDS issue | HND `+0.312%`, NHD `+0.239%`, below the `0.5%` local gate | Rejected |
| Progressive PV LDS issue | HND `-3.165%` | Rejected |
| Hoist D64 V global loads before the barrier | `-0.349%` paired geomean | Rejected |
| DPP row-share alpha fan-out | HND `-0.743%`, NHD `-0.602%` despite cleaner ISA | Rejected |
| Temporary FP16 score exponentials | LLVM retained FP32 exponentials and added 64 conversions | Rejected before timing |
| Half-up E5M2 Q encoding | Overall `-0.176%`; work removed only once per query tile | Rejected; retain RNE |
| D64 NHD `Br=256,Bc=64` | Two-shape geomean `-1.889%` | Rejected |
| D64 HND `Br=128,Bc=128` | 242 VGPRs, 50 above the hard gate | Rejected before timing |
| Tile-local FP16 PV WMMA buffer | Correct at existing tolerances, but HND geomean `-8.739%` and near-192-VGPR pressure | Rejected |
| D128 decoded-Q cache or double buffering | D128 already uses 191 VGPRs and the full 32 KiB LDS budget | Not admitted without prior resource reduction |

## Remaining Work

The forward path is closed for production optimization until new profiling identifies a material opportunity. Remaining work is maintenance or explicitly deferred scope:
- Keep the `168/168` public contract and 36-row AITER matrix as regression gates for any shared host, ABI, compiler, or CK Tile change.
- Revisit D128 only if a source change first lowers the loaded image below the current 191-VGPR or 32-KiB limit. Do not add persistent state to the current image.
- Admit a local schedule change only with at least a repeatable `0.5%` target-domain geomean gain, majority wins, no regression above `1%`, unchanged correctness, and no resource regression.
- Admit a new tile shape or launch policy only with at least a `5%` target-domain geomean gain and no effect on non-target dispatches.
- Treat causal attention, masks, dropout, bias, GQA/MQA, additional dtypes, and wide-index kernels as separate feature campaigns rather than extensions of the current optimized path.

Hard resource gates remain 192 allocated VGPRs, 32,768 bytes LDS, zero private/scratch memory, zero spills, and no material LDS-bank-conflict regression.

## Verification and Artifacts

Primary repository checks:
- `test_attn_hip.py`: `168/168` public-contract cases.
- `benchmark_attn_hip.py`: AITER/Feather benchmark harness.
- `kernel_attn/hip/build/attn_hip_ext/build.ninja`: current independent forward translation units.

Authoritative artifacts:
- Final matrix: `~/tmp/feather_attn/phase11_final/matrix/attn.csv`.
- Final loaded metadata: `~/tmp/feather_attn/phase11_final/metadata.txt`.
- Final LDS conflicts: `~/tmp/feather_attn/phase11_final/lds_bank_conflicts/summary.json`.
- D64 HND cache timing/profile: `~/tmp/feather_attn/phase11b_qcache_full_paired_robust.json` and `phase11b_qcache_profiles/`.
- D64 NHD strided timing/profile: `~/tmp/feather_attn/phase11c_selected_domain_paired.json` and `phase11c_strided_profiles/`.
- Broader AITER/Feather counter review: `~/tmp/feather_attn/review/counter_matrix_summary.txt`.

Historical experiment details remain in the `~/tmp/feather_attn/phase*` logs. They are evidence for the accepted/rejected table, not the current implementation roadmap.
