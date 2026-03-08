# gfx1151 Reference

This file keeps hardware/runtime/profiling facts for gfx1151 (this machine).

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

Besides VGPR capacity, occupancy is also bounded by LDS capacity.

## WMMA Facts

- Tile shape: `16x16 @ 16x16`
- Instruction latency: `32 cycles`
- Supported families include fp16/bf16 and i8/u8 WMMA variants.

## rocprofv3 Workflow

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

## PC Sampling

PC sampling is available on gfx1151 using a custom-built amdgpu driver, ROCr, and ROCProfiler.

### Running PC Sampling

Use `$ROCM_PATH/bin/rocprofv3` directly (not the venv wrapper, which loads stock libraries without GFX11 PC sampling support). Both `host_trap` and `stochastic` methods are supported. Stochastic provides precise zero-skid instruction sampling.

**For Host-Trap (Time-based):**
```bash
$ROCM_PATH/bin/rocprofv3 \
  --pc-sampling-method host_trap \
  --pc-sampling-unit time \
  --pc-sampling-interval 5000 \
  -o <output_prefix> \
  -- python <script>.py
```
- `--pc-sampling-interval 5000`: scan interval in microseconds (5ms).

**For Stochastic (Cycle-based):**
```bash
$ROCM_PATH/bin/rocprofv3 \
  --pc-sampling-method stochastic \
  --pc-sampling-unit cycles \
  --pc-sampling-interval 1048576 \
  -o <output_prefix> \
  -- python <script>.py
```
- **Important:** GFX11.5 hardware stochastic sampling *only* supports `cycles` (not `time`), and the interval *must* be a power of 2 (e.g., `1048576` cycles). Using `time` or non-power-of-2 intervals will result in a "configuration is not supported" error.
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

At `--pc-sampling-interval 5000` with 200 iterations of an N=8192 matmul
(~7s wall time):
- Old SQ_IND driver: ~570K samples (reads all active waves per scan)
- Host-trap driver (mainline): ~10K samples (traps one wave per SIMD/slot per SQ_CMD)

### Kernel dmesg Verification

```bash
dmesg | grep pcs
# Expected: "pcs: thread started interval_us=5000 vmid=N"
# Expected: "pcs: thread exiting, sent NNN traps"
```

### Overlap Measurement (Alternative)

When PC sampling is not available (e.g. stock driver), overlap can be
estimated by controlled mode decomposition (`full`, `no_overlap`,
`comm_only`, `comp_only`). See experiment P6-A/P6-B in the e5m2
optimization plan for details.

## Thread Tracing

Thread Tracing captures per-wave instruction execution timelines. It is supported on gfx1151 and does not require a custom kernel (works with mainline driver).

**Running Thread Tracing:**
```bash
$ROCM_PATH/bin/rocprofv3 --att -d <output_dir> -o <prefix> -- python <script>.py
```
*Note: By default, it only traces the FIRST dispatch of each kernel. If your script loops over the kernel multiple times, only the first one is captured, so you can reduce `N_ITER` to save time. If the target kernel is not the first dispatch (e.g. initialization/random generation kernels run first), check `ui_output_agent_*/code.json` in each directory to find the target kernel trace.*

**Outputs:**
- `stats_*.csv`: Aggregated latency, stall, and idle cycle counts for every instruction in the kernel.
- `*.att`: Raw SQTT binary trace data.
- `ui_output_agent_*/`: Directory containing `wave_*.json` and `code.json` which map out the exact cycle-by-cycle execution of every wave.

### Analyzing Thread Trace Data

You can load the `.att` file or the `ui_output_agent_*/` directory into the **ROCprof Compute Viewer** to visualize the overlap between memory operations (LDS/Global) and compute (WMMA) on a timeline.

Alternatively, you can write python scripts using `matplotlib` and `pandas` to programmatically plot the timeline directly from the `wave_*.json` and `code.json` pairs.

**Important Pitfalls to Remember:**
1. **Single-Wave Perspective:** A trace timeline only plots the execution of *one specific wave* on *one specific SIMD*. If you see a massive `Sync` stall (e.g. `s_waitcnt vmcnt`), that single wave is indeed completely stalled.
2. **Macro-Level Hiding:** Do not confuse a single-wave stall with global GPU starvation. Even if the traced wave shows it is stalled 50% of the time, the *overall* GPU TFLOPS might be hitting 75%+ of theoretical max. This indicates the massive `Sync` blocks are being successfully hidden by the hardware's macro-level scheduler staggering the memory requests across the other 79 SIMDs on the chip.
3. **Internal Bottleneck:** To find the true *internal* kernel bottleneck, calculate the ratio of instructions executed *within the active math loop phase* (ignoring the global wait stalls).
