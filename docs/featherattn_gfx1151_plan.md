# FeatherAttn gfx1151 Forward Kernel

## Status

The gfx1151 forward kernel is production-qualified for dense non-causal FP16 attention with contiguous HND and NHD tensors and head dimensions 64 and 128. The accepted kernel body was qualified at commit `01454e3`; the later source cleanup changed file ownership and build organization but did not change forward arithmetic or dispatch.

The current result is:
- `168/168` public-contract cases pass.
- The complete 36-row performance matrix is shown below, with no aggregate or representative-row summary replacing individual shapes.
- All 20 qualified forward images stay at or below 191 used VGPRs, 32,768 bytes LDS, zero private memory, and zero SGPR/VGPR spills.
- A disposable packed-RNE encoding candidate passed the public contract and resource gates, but did not produce a repeatable complete-kernel speedup. It was not promoted.
- A disposable truncation candidate reduced code further but failed the attention numerical gate. It is closed.
- No production optimization is currently open. Any reopening must follow the ranked plan and promotion gates below.

Hardware and execution assumptions:

| Item | Value |
| --- | --- |
| GPU | Radeon 8060S, gfx1151 |
| Compute units | 40 CUs, 20 WGPs |
| Wave size | 32 |
| Main comparison | FlashAttention AITER Triton `attn_fwd` |
| Arithmetic | FP16 WMMA with FP32 score and output accumulation |

## Investigation Closeout

The final exploratory question was whether the Q FP16-to-E5M2 path could remove RNE work or use explicit packed operations without changing the production contract.

| Candidate | Numerical result | Linked resource result | Timing result | Decision |
| --- | --- | --- | --- | --- |
| Packed RNE encoding | `168/168` public cases; bitwise identical to the scalar-RNE aligned image in the focused comparison | Same 20-image resource envelope; D64/D128 stayed at the existing VGPR and LDS values | 12 aligned paired rows: `0.9969x` geometric mean, 4/12 wins, range `0.9828x-1.0063x` | Promising ISA experiment, no promotion |
| Naive E5M2 truncation | Exhaustive 65,536-pattern conversion probe had zero GPU-reference mismatches, but attention tests reached `Rel-L2` about `0.122-0.125` and failed elementwise cases | No resource issue in the disposable aligned image | Not relevant after numerical failure | Closed |
| Half-up alternative | Finite and contract-qualified in the prior campaign, but not output-exact and had no repeatable timing gain | No useful allocation transition | Prior robust result was `-0.176%` overall | Closed; retain RNE |

The packed-RNE candidate was compiled with the recovered production-equivalent command, linked with all aligned, query-tail, KV-tail, combined-tail, and D64 NHD strided translation units, and exercised through the host selector. Its ISA reduction is real: in representative HND symbols, D64 changed from 8,224 to 7,500 code bytes and from 1,250 to 1,158 encoded instructions; D128 changed from 13,104 to 11,620 bytes and from 1,949 to 1,739 instructions. Scalar `v_lshrrev_b16` and `v_add_nc_u16` packing operations were replaced by packed 16-bit operations and byte permutes. The work is paid once per query tile and is therefore mostly amortized by the KV loop, which explains the neutral complete-kernel timing.

The truncation probe is useful for understanding conversion semantics, not for promotion. It preserved signs and did not increase finite magnitudes, but 510 of 2,046 subnormal inputs became zero and 510 of 2,046 NaN inputs became infinity. Removing RNE is consequently a numerical-policy change, not an instruction substitution.

The screening timing used the frozen phase-11c aligned baseline and 20 alternating samples per row. It did not replace the authoritative 36-row matrix or exercise a production selector change. Production source, ABI, dispatch, and registration remain unchanged.

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

The host selector is `SelectLauncher(int64_t head_dim, bool nhd, bool pad_q, bool pad_kv)`. It selects the D64 or D128 HND/NHD entry point from the query-tail and KV-tail flags; aligned, query-tail, KV-tail, and combined-tail paths are distinct symbols. The separate D64 NHD strided selector is used only after the long-sequence grouping policy admits partition-aware launches. This selector topology and the raw ABI are part of the qualified contract.

HND uses flattened batch/head ownership. NHD makes head the fastest grid axis so adjacent workgroups touch adjacent head offsets. Two bounded host policies handle long NHD tensors:
- D128 may split heads into sequential LLC-sized groups when the total K/V working set is large enough to benefit.
- D64 may use the strided physical-head mapping `group_index + local_head * group_count`. The selector avoids group counts divisible by eight because those strides recreate the gfx1151 memory-partition cliff.

All sublaunches run on the caller's current stream. Tail selection and grouping do not change the tensor layout or require a physical transpose.

### Q Decode Caching

D64 HND has enough register headroom to cache decoded Q fragments across KV tiles. Aligned, query-tail, and combined-tail images cache all four D16 fragments; the KV-tail image caches three. D64 NHD and all D128 images retain decode-on-use because their measured tradeoffs or resource pressure do not justify persistent decoded Q state.

### E5M2 Conversion and Packing

The numerical baseline is RNE. The current scalar encoder is equivalent to:

```text
rounded = fp16_bits + 0x007f + ((fp16_bits >> 8) & 1)
e5m2 = rounded >> 8
```

The decode path already loads packed bytes and uses `v_perm_b32`. gfx1151 has no native FP8 WMMA path, so Q quantization and decode are VALU/frontend work around FP16 WMMA; prior tracing attributed roughly 23% of math-pipeline execution to unpacking. Explicit `uint32` packing with `0x07050301` byte-permute control is the strongest low-risk instruction-level lead found so far, but its setup cost is too small relative to the KV loop to justify a standalone production change.

The packed candidate must be treated as an integrated kernel candidate. A standalone pack microbenchmark measured `0.0563 ms` versus `0.0974 ms` for scalar packing, but the complete aligned kernel did not reproduce that gain. Future work should only revisit packing when it is fused with Q load/scale and also shortens live state or creates an occupancy transition.

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
| D64 NHD, base and strided | 130-162 | 16,384 B | 0 / 0 | Includes four partition-aware images; the combined-tail image is the upper end |
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

### D64 Results

Every layout, head count, and sequence length in the authoritative matrix is listed here. `Feather / AITER` is the throughput ratio; values above `1.000x` favor FeatherAttn.

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 4096 | 33.981 | 35.898 | 1.056x |
| HND | 16 | 8192 | 32.144 | 33.999 | 1.058x |
| HND | 16 | 16384 | 31.747 | 33.580 | 1.058x |
| HND | 32 | 4096 | 32.698 | 34.870 | 1.066x |
| HND | 32 | 8192 | 31.047 | 33.793 | 1.088x |
| HND | 32 | 16384 | 29.395 | 33.654 | 1.145x |
| HND | 56 | 4096 | 29.749 | 33.776 | 1.135x |
| HND | 56 | 8192 | 28.972 | 33.386 | 1.152x |
| HND | 56 | 16384 | 29.140 | 33.761 | 1.159x |
| NHD | 16 | 4096 | 32.908 | 32.522 | 0.988x |
| NHD | 16 | 8192 | 32.007 | 31.376 | 0.980x |
| NHD | 16 | 16384 | 31.943 | 31.197 | 0.977x |
| NHD | 32 | 4096 | 31.037 | 31.258 | 1.007x |
| NHD | 32 | 8192 | 28.596 | 29.960 | 1.048x |
| NHD | 32 | 16384 | 22.754 | 29.088 | 1.278x |
| NHD | 56 | 4096 | 29.068 | 30.820 | 1.060x |
| NHD | 56 | 8192 | 28.092 | 29.749 | 1.059x |
| NHD | 56 | 16384 | 28.228 | 29.959 | 1.061x |

### D128 Results

| Layout | H | N | AITER TFLOPS | Feather TFLOPS | Feather / AITER |
| --- | ---: | ---: | ---: | ---: | ---: |
| HND | 16 | 4096 | 35.015 | 34.262 | 0.978x |
| HND | 16 | 8192 | 32.737 | 33.444 | 1.022x |
| HND | 16 | 16384 | 31.653 | 34.120 | 1.078x |
| HND | 32 | 4096 | 31.800 | 34.731 | 1.092x |
| HND | 32 | 8192 | 30.872 | 33.641 | 1.090x |
| HND | 32 | 16384 | 30.822 | 34.419 | 1.117x |
| HND | 56 | 4096 | 28.927 | 34.007 | 1.176x |
| HND | 56 | 8192 | 30.109 | 33.727 | 1.120x |
| HND | 56 | 16384 | 30.628 | 34.372 | 1.122x |
| NHD | 16 | 4096 | 31.842 | 31.202 | 0.980x |
| NHD | 16 | 8192 | 30.660 | 30.811 | 1.005x |
| NHD | 16 | 16384 | 24.062 | 30.240 | 1.257x |
| NHD | 32 | 4096 | 26.009 | 30.367 | 1.168x |
| NHD | 32 | 8192 | 21.718 | 30.168 | 1.389x |
| NHD | 32 | 16384 | 21.566 | 27.393 | 1.270x |
| NHD | 56 | 4096 | 26.901 | 30.890 | 1.148x |
| NHD | 56 | 8192 | 27.655 | 31.598 | 1.143x |
| NHD | 56 | 16384 | 28.485 | 31.454 | 1.104x |

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
| Packed RNE Q encoding | Representative HND code size fell 8.8% at D64 and 11.3% at D128, but the 12-row aligned screening geomean was `0.9969x` with no resource transition | Keep as a future fused-frontend lead, not a standalone change |
| Naive E5M2 truncation | Attention numerical failures despite a clean exhaustive bit-conversion probe; it changes subnormal and NaN behavior | Closed unless a separately specified numerical policy is approved |

## Ranked Forward Plan

The production path remains unchanged. The following are the only currently promising directions, ordered by expected leverage and evidence quality:
- Lower D128 live state before adding persistent state. D128 is pinned at 191 VGPRs and 32 KiB LDS, so decoded-Q caching, double buffering, or a larger tile cannot be admitted directly. Inspect source-to-MIR-to-ISA liveness around Q decode, FP8 unpack, WMMA fragments, and the output epilogue. A candidate is interesting only if it creates a natural allocation or occupancy transition without private memory or spills.
- Fuse packed RNE encoding into the Q frontend. The packed-RNE experiment materially reduced symbol bytes and scalar packing instructions while preserving bitwise output. Its complete-kernel geomean did not improve because the work is once per query tile. Reopen it only as part of a load/scale/decode schedule that reduces live values or exposes useful overlap; do not add a helper launch or a standalone conversion pass.
- Maintain and, only with new evidence, tune long-NHD grouping. Partition-aware D64 NHD mapping and bounded D128 LLC grouping are the largest measured forward wins. Threshold or group-count changes are more promising than generic LDS or DRAM work, but they must be evaluated through the existing selector and complete matrix because they affect launch count and cache residency.
- Defer new tile shapes and local schedule changes. Existing profiles show low LDS conflict cost and the prior local schedule experiments were neutral or negative. Reopen this category only after a new symbol-matched counter or MIR/ISA result identifies a specific dependency or occupancy transition.

### Promotion Gates

Every candidate must satisfy all gates before any production source or selector change:
- The repository public fixture remains `168/168`, including HND/NHD, D64/D128, independent tails, odd heads, and batch two. Outputs must be finite; directional screening requires cosine at least `0.997`, with `Rel-L2 <= 0.10` preferred and `< 0.20` exploratory, in addition to the public elementwise envelope.
- Every affected linked gfx1151 image stays within 192 campaign-rounded VGPRs, 32,768 bytes LDS, zero private/scratch memory, and zero spills. Inspect the exact runtime symbol, metadata, ISA, and code-object hash.
- The raw ABI, operator IDs, selector behavior, current-device validation, and current-stream execution remain unchanged unless the candidate is explicitly a host-policy experiment.
- Timing uses complete execution, including all helper launches, conversions, synchronization, and selector sublaunches. A local change needs a repeatable at least `0.5%` target-domain geomean gain, majority wins, and no regression above `1%`. The full 36-row matrix remains the final regression gate.

### Closed Directions

RNE removal by truncation, half-up conversion, forced register allocation, generic bandwidth/cache claims, and direct FP8 compression of persistent attention state are closed for the current design. Causal attention, masks, dropout, bias, GQA/MQA, additional dtypes, and wide-index kernels remain separate feature campaigns.

Hard resource gates remain 192 allocated VGPRs, 32,768 bytes LDS, zero private/scratch memory, zero spills, and no material LDS-bank-conflict regression.

## Verification and Artifacts

Primary repository checks:
- `test_attn_hip.py`: `168/168` public-contract cases.
- `benchmark_attn_hip.py`: AITER/Feather benchmark harness.
- `kernel_attn/hip/build/attn_hip_ext/build.ninja`: current independent forward translation units.
- `~/tmp/feather_attn/fwd_reopen_20260816/`: disposable packed-RNE and truncation sources, linked images, metadata, ISA summaries, contract output, and focused timing.
- `~/tmp/feather_attn/e5m2_truncation_campaign/e5m2_exhaustive.json`: exhaustive conversion probe.
- `~/tmp/feather_attn/e5m2_truncation_campaign/pack_benchmark.json`: standalone scalar versus permute pack probe.
- `~/tmp/feather_attn/phase11e4_half_up_robust.json`: prior half-up timing and output evidence.

Authoritative artifacts:
- Final matrix: `~/tmp/feather_attn/phase11_final/matrix/attn.csv`.
- Final loaded metadata: `~/tmp/feather_attn/phase11_final/metadata.txt`.
- Final LDS conflicts: `~/tmp/feather_attn/phase11_final/lds_bank_conflicts/summary.json`.
- D64 HND cache timing/profile: `~/tmp/feather_attn/phase11b_qcache_full_paired_robust.json` and `phase11b_qcache_profiles/`.
- D64 NHD strided timing/profile: `~/tmp/feather_attn/phase11c_selected_domain_paired.json` and `phase11c_strided_profiles/`.
- Broader AITER/Feather counter review: `~/tmp/feather_attn/review/counter_matrix_summary.txt`.

The disposable forward reopen used `~/venv_torch/lib/python3.14/site-packages/_rocm_sdk_devel/bin/hipcc` with `-O3 -fno-gpu-rdc --offload-arch=gfx1151`, recovered from Ninja. Candidate artifacts are evidence only; they are not production build inputs. Historical experiment details remain in the `~/tmp/feather_attn/phase*` logs. They are evidence for the accepted/rejected table, not the current implementation roadmap.
