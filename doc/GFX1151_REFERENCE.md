# GFX1151 Reference

This file keeps hardware/profiling/runtime facts for this machine.

## Hardware Snapshot

| Item | Value |
|---|---|
| GPU | RDNA 3.5 (`gfx1151`) |
| Compute Units | 40 |
| Wave Size | 32 |
| Max Clock | 2900 MHz |
| LDS per workgroup | 64 KB |
| Memory | LPDDR5 256-bit 8000 MT/s |
| Theoretical BW | 256 GB/s |
| Practical BW | ~200 GB/s |

Theoretical fp16 compute: 40 CUs * 2 SIMD units/CU * 32 Vector ALUs/SIMD unit * 2 (VOPD dual issue or WMMA) * 2 (fp16 packing) * 2 (fused multiply-add) * 2.9 GHz = 59.4 TFLOPS

## Occupancy Quick Math

Per-SIMD limits:
- Max waves/SIMD: `16`
- VGPR budget/SIMD: `1536`

Rule of thumb:
```
waves_by_vgpr = floor(1536 / vgpr_per_wave)
occupancy_per_simd = min(waves_by_vgpr, 16) / 16
```

Example: `176 VGPR` -> `floor(1536/176)=8 waves` -> `50%` per SIMD.

## WMMA Facts (gfx1151)

- Tile shape: `16x16 @ 16x16`
- Instruction latency: `32 cycles`
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

## PC Sampling on gfx1151

PC sampling is available on gfx1151 using a custom-built amdgpu driver, ROCr, and ROCProfiler.

### Running PC Sampling

Use `$ROCM_PATH/bin/rocprofv3` directly (not the venv wrapper, which loads
stock libraries without gfx11 PC sampling support):

```bash
$ROCM_PATH/bin/rocprofv3 \
  --pc-sampling-method host_trap \
  --pc-sampling-unit time \
  --pc-sampling-interval 5000 \
  -o <output_prefix> \
  -- python <script>.py
```

- `--pc-sampling-interval 5000`: scan interval in microseconds (5ms).
- Default output is SQLite (rocpd format) with a `rocpd_pc_sampling` table
  containing columns: `timestamp`, `exec_mask`, `dispatch_id`, `instruction`,
  `instruction_comment`, `correlation_id`.
- Add `-f csv` for CSV output instead (columns match the DB schema).
- rocprofv3 decodes PCs to instruction text directly (no manual
  PC-to-disassembly mapping needed).

### Querying the DB

```bash
# Sample count
sqlite3 results.db "SELECT COUNT(*) FROM rocpd_pc_sampling;"

# Category breakdown
sqlite3 results.db "
SELECT
  CASE
    WHEN instruction LIKE 'ds_load%' THEN 'LDS Read'
    WHEN instruction LIKE 'ds_store%' THEN 'LDS Write'
    WHEN instruction LIKE 'global_load%' THEN 'Global Load'
    WHEN instruction LIKE 'global_store%' THEN 'Global Store'
    WHEN instruction LIKE '%wmma%' THEN 'WMMA'
    WHEN instruction LIKE 'v_perm_b32%' OR instruction LIKE 'v_pk_%'
         OR instruction LIKE 'v_cvt_%' THEN 'FP Convert'
    WHEN instruction LIKE 's_barrier%' OR instruction LIKE 's_waitcnt%'
         OR instruction LIKE 'buffer_gl0_inv%' THEN 'Sync'
    WHEN instruction LIKE 's_%' THEN 'SALU'
    WHEN instruction LIKE 'v_%' THEN 'VALU Other'
    ELSE 'Other'
  END AS category,
  COUNT(*) AS cnt,
  ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM rocpd_pc_sampling), 1) AS pct
FROM rocpd_pc_sampling GROUP BY category ORDER BY cnt DESC;
"

# Top instructions
sqlite3 results.db "
SELECT instruction, COUNT(*) AS cnt
FROM rocpd_pc_sampling GROUP BY instruction ORDER BY cnt DESC LIMIT 20;
"
```

### Typical Sample Counts

At `--pc-sampling-interval 5000` with 200 iterations of an N=8192 GEMM
(~7s wall time), expect ~570K samples.

### Kernel dmesg Verification

```bash
dmesg | grep pcs
# Expected: "pcs: thread started interval_us=5000 ..."
# Expected: "pcs: thread exiting, total_delivered=NNNNNN loops=NNN"
```

### Overlap Measurement (Alternative)

When PC sampling is not available (e.g. stock driver), overlap can be
estimated by controlled mode decomposition (`full`, `no_overlap`,
`comm_only`, `comp_only`). See experiment P6-A/P6-B in the e5m2
optimization plan for details.
