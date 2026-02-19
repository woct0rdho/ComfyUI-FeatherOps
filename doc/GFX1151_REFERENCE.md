# GFX1151 Reference

This file keeps hardware/profiling/runtime facts for this machine.

## Hardware Snapshot

| Item | Value |
|---|---|
| GPU | RDNA 3.5 (`gfx1151`) |
| Compute Units | 40 |
| Wave Size | 32 only |
| Max Clock | 2900 MHz |
| LDS per workgroup | 64 KB max |
| Memory | LPDDR5 256-bit 8000 MT/s |
| Theoretical BW | 256 GB/s |
| Practical BW | ~200 GB/s |

Theoretical fp16 compute: 40 CUs * 2 SIMD units/CU * 32 Vector ALUs/SIMD unit * 2 (VOPD dual issue or WMMA) * 2 (fp16 packing) * 2 (fused multiply-add) * 2.9 GHz = 59.4 TFLOPS

## Occupancy Quick Math

Per-SIMD limits:
- Max waves/SIMD: `16`
- VGPR budget/SIMD: `1536`

Rule of thumb:
```text
waves_by_vgpr = floor(1536 / vgpr_per_wave)
occupancy_per_simd = min(waves_by_vgpr, 16) / 16
```

Example: `176 VGPR` -> `floor(1536/176)=8 waves` -> `50%` per SIMD.

## WMMA Facts (gfx1151)

- Tile shape: `16x16x16`
- Rough instruction latency: `16 cycles`
- Throughput target: ~1 WMMA op / 16 cycles / SIMD
- Supported families include fp16/bf16 and i8/u8 WMMA variants.

## rocprofv3 Workflow (Known Good)

Basic profiling:
```bash
rocprofv3 --kernel-trace --stats -d <out_dir> -o <prefix> -- python <script>.py
```

Single-PMC run (safer on gfx1151):
```bash
rocprofv3 --kernel-trace --stats --pmc LDSBankConflict -d <out_dir> -o <prefix> -- python <script>.py
```

Read DB quickly:
```bash
sqlite3 <db> ".tables"
sqlite3 <db> "SELECT * FROM top_kernels"
sqlite3 <db> "SELECT name, duration/1000.0 AS us, vgpr_count, sgpr_count, lds_size, grid_x, workgroup_x FROM kernels"
```

Known usable counters in this project:
- `LDSBankConflict`
- `L2CacheHit`
- `VALUInsts`

Known pitfalls:
- Error 38 for unsupported/overpacked counter sets.
- JIT and lock-file issues can make profiling look hung.
- Profile overhead changes wall-time; use benchmark scripts for performance decisions.

## N=8192 Runtime Facts from Existing Artifacts

Data sources:
- `rocprof_torch_8192_steady/torch8192_steady_results.db`
- `rocprof_hip_8192/hip8192_results.db`
- `rocprof_hip_8192_matched/hip8192_matched_results.db`

Measured main kernels:

| Kernel | Avg time (ms) | VGPR | SGPR | LDS (B) | Grid | Workgroup |
|---|---:|---:|---:|---:|---|---|
| `torch.compile` GEMM | 40.186 | 256 | 128 | 17408 | (8192, 64) | (128, 1) |
| HIP (before launch-shape match) | 40.272 | 192 | 128 | 16896 | (4096, 128) | (64, 2) |
| HIP (after launch-shape match) | 41.082 | 192 | 128 | 16896 | (8192, 64) | (128, 1) |

Occupancy implication for this tile:
- `128` threads/WG = `4` wave32/WG.
- LDS-limited WGs/CU:
  - torch: `floor(65536/17408)=3`
  - hip: `floor(65536/16896)=3`
- So both sit at `3 WG/CU = 12 waves/CU = 37.5%` CU wave occupancy.

Conclusion:
- For this case, occupancy is LDS-limited, not VGPR-limited.
- Matching launch geometry alone does not close the gap.

## Overlap Measurement Without PC Sampling

PC sampling is unavailable on gfx1151, so overlap can be estimated by controlled mode decomposition:

- `full`
- `no_overlap`
- `comm_only`
- `comp_only`

Derived metrics:
- direct gain: `T_no_overlap - T_full`
- decomposition gain: `T_comm_only + T_comp_only - T_full`

Observed at `N=8192`:
- config `2,2,2,2,4,4`: near-zero/negative overlap gain.
- config `2,2,2,4,4,4`: positive overlap metric but slower total kernel.

Practical rule:
- optimize total runtime first; overlap percentage alone is not the objective.
