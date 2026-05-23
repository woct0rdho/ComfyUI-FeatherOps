# gfx1151 Reference

This file keeps hardware/runtime/profiling facts for gfx1151 (this machine).

## Hardware Facts

- ISA: RDNA3.5
- Compute units: 40 (20 work-group processors)
- Wave size: 32
- Max clock: 2900 MHz
- LDS per CU: 64 KB (128 KB per WGP)
- VRAM: LPDDR5 256-bit 8000 MT/s

Theoretical fp16 compute: 40 CUs * 2 SIMD units per CU * 32 Vector ALUs per SIMD unit * 2 (VOPD dual issue or WMMA) * 2 (fp16 packing) * 2 (fused multiply-add) * 2.9 GHz = 59.4 TFLOPS

Theoretical VRAM bandwidth: 256 bits * 8000 MT/s = 256 GB/s

## Occupancy Calculation

Per-SIMD limits:
- Max waves per SIMD: 16
- VGPR budget per SIMD: 1536

1 VGPR consumes 4 bytes per thread. Since a wave has 32 threads, 1 VGPR consumes 128 bytes of register file space. 1536 VGPRs = 192 KB register file space per SIMD.

The hardware limit for waves per workgroup is 32 (1024 threads per workgroup). Common choices of waves per workgroup are 4 or 8.

```
# VGPRs are allocated in blocks of 24 (for wave32).
vgpr_allocated_per_wave = ceil(vgpr_used_per_wave / 24) * 24
waves_by_vgpr = floor(1536 / vgpr_allocated_per_wave)

# A WGP has 4 SIMDs. LDS is shared across the WGP.
workgroups_by_lds = floor(131072 / lds_per_workgroup_in_bytes)
waves_by_lds_per_simd = floor(workgroups_by_lds * waves_per_workgroup / 4)

occupancy_per_simd = min(waves_by_vgpr, waves_by_lds_per_simd, 16) / 16
```

Examples:
- `192 VGPR allocated per wave` -> `floor(1536/192) = 8 waves per SIMD` -> `50%` occupancy limit by VGPR.
- `65536 bytes LDS per workgroup` and `4 waves per workgroup` -> `floor(131072/65536) = 2 workgroups per WGP` -> `floor(2 * 4 / 4) = 2 waves per SIMD` -> `12.5%` occupancy limit by LDS.

Note on LDS capacity: A WGP has **128 KB** of physical LDS (64 KB per CU). However, the architecture restricts a single workgroup to allocating a maximum of **64 KB**. Therefore, tools like rocminfo and sysfs will report 64 KB (the software allocation limit per-workgroup, which corresponds to the per-CU physical size), but a WGP can physically fit two such 64 KB workgroups simultaneously. Occupancy bottlenecks on LDS only when the combined LDS requests of all active workgroups exceed the 128 KB per-WGP limit.

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

Verified multi-counter run for this project:
```bash
rocprofv3 --kernel-trace --stats \
  --pmc L2CacheHit VALUInsts LDSBankConflict \
  -d <out_dir> -o <prefix> -- python <script>.py
```

To restrict output to the target kernel, use a kernel filter:
```bash
rocprofv3 --kernel-trace --stats \
  --pmc LDSBankConflict \
  --kernel-include-regex <kernel_regex> \
  -d <out_dir> -o <prefix> -- python <script>.py
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

Pitfalls:
- Error 38 for unsupported/overpacked counter sets.
- JIT and lock-file issues can make profiling look hung.
- Profile overhead changes wall-time. Use benchmark scripts for performance decisions.
- If a counter-collection run produces no DB, first verify the output directory was actually created. On gfx1151, the above `L2CacheHit + VALUInsts + LDSBankConflict` combination does work.

## PC Sampling

PC sampling is available on gfx1151 using a custom-built amdgpu driver, ROCr, and ROCProfiler, see https://github.com/woct0rdho/linux-amdgpu-driver

### Running PC Sampling

Use `$ROCM_PATH/bin/rocprofv3`, not the one in the venv `bin/`, and not the one in `~/rocm-systems/build/`, to ensure all libraries are correctly loaded. Both `host_trap` and `stochastic` methods are supported. Stochastic provides precise zero-skid instruction sampling, while host-trap is software-driven and can have sampling skid.

For Host-Trap (Time-based):
```bash
$ROCM_PATH/bin/rocprofv3 \
  --pc-sampling-method host_trap \
  --pc-sampling-unit time \
  --pc-sampling-interval 5000 \
  -d <out_dir> -o <prefix> -- python <script>.py
```

- `--pc-sampling-interval 5000`: scan interval in microseconds (5ms).

For Stochastic (Cycle-based):
```bash
$ROCM_PATH/bin/rocprofv3 \
  --pc-sampling-method stochastic \
  --pc-sampling-unit cycles \
  --pc-sampling-interval 1048576 \
  -d <out_dir> -o <prefix> -- python <script>.py
```

- GFX11.5 hardware stochastic sampling only supports `cycles` (not `time`), and the interval must be a power of 2 (e.g., `1048576` cycles). Using `time` or non-power-of-2 intervals will result in a "configuration is not supported" error.
- Default output is SQLite (rocpd format). Current rocpd schema version remains `4` for internal PC-sampling output.
- `rocpd_pc_sampling` contains common sample columns plus `method`: `method`, `timestamp`, `exec_mask`, `dispatch_id`, `instruction`, `instruction_comment`, `correlation_id`.
- `method` is either `host_trap` or `stochastic`, so mixed-method databases are unambiguous.
- Stochastic rocpd rows also include the same decoded fields exposed by stochastic CSV: `wave_issued_instruction`, `instruction_type`, `stall_reason`, and `wave_count`. These columns are null for host-trap rows.
- Add `-f csv` for CSV output when a flat text artifact is easier to inspect.
- rocprofv3 decodes PCs to instruction text directly (no manual PC-to-disassembly mapping needed).
- gfx1151 stochastic metadata uses the local KFD IOCTL v1.5/gc12-style layout: raw instruction type comes from `SQ_PERF_SNAPSHOT_DATA[5:2]`, stall reason from `[8:6]`, sampling-lock error from bit 14, wave count from `SQ_PERF_SNAPSHOT_DATA1[5:0]`, and arbiter issue/stall state from `DATA1[24:9]`.
- Conditional branch samples preserve the raw hardware branch taken/not-taken outcome. Decoded `s_cbranch*` text alone should not be treated as proof that the branch was taken.

### Interpreting PC Sampling Data

For stochastic PC sampling, a critical architectural detail is that it records the instruction the program counter (PC) is currently pointing to, which is the instruction **waiting to be issued**. It does not necessarily record instructions currently executing in the pipeline. Host-trap PC sampling can also report nearby instructions because it is interrupt/skid based.

If an instruction has a high sample count, it means the sequencer spent a long time stalled trying to issue that instruction, not necessarily that the instruction took a long time to compute.

Examples:
- Instruction Fetch Stalls (e.g., `v_perm_b32`): A `v_wmma` executes in 32 cycles, while a `v_perm_b32` executes in 1 cycle. However, if you use inline 32-bit literal constants (e.g., `0x30c020c`), the `v_perm_b32` becomes a massive 96-bit (3 DWORD) instruction. When the sequencer tries to fetch these massive instructions for multiple concurrent waves, it chokes the instruction fetch frontend. The PC freezes, pointing at the `v_perm` instruction for dozens of cycles waiting for instruction memory. Thus, `v_perm_b32` will incorrectly appear to take more "time" than `v_wmma` in the PC trace. It's actually an instruction fetch stall.
- Structural Queue Stalls (e.g., `ds_load_b128`): A wave issuing `ds_load_b128` requests 512 bytes of LDS data. The LDS unit can process 128 bytes/cycle, so one load takes 4 cycles. If 8 resident waves constantly fire these massive loads, the internal LDS memory instruction queue becomes fully saturated. The sequencer is structurally blocked from issuing further LDS instructions. During this wait, the PC is frozen pointing at the blocked `ds_load`, resulting in massive sample counts. This is a queue-full stall, not execution time.

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

At `--pc-sampling-interval 5000` with 200 iterations of an N=8192 matmul (~7s wall time):
- Old `SQ_IND` driver: ~570K samples (reads all active waves per scan)
- Host-trap driver (mainline): ~10K samples (traps one wave per SIMD/slot per `SQ_CMD`)

### Optional Kernel dmesg Verification

Some debug driver builds log PC-sampling worker activity. Production or quieter builds may not emit these lines.

```bash
dmesg | grep pcs
# Possible debug output: "pcs: thread started interval_us=5000 vmid=N"
# Possible debug output: "pcs: thread exiting, sent NNN traps"
```

### Overlap Measurement (Alternative)

When PC sampling is not available (e.g. stock driver), overlap can be estimated by controlled mode decomposition (`full`, `no_overlap`, `comm_only`, `comp_only`). See experiment P6-A/P6-B in the e5m2 optimization plan for details.

## Thread Tracing

Thread tracing captures per-wave instruction execution timelines. The current gfx1151 support is in userspace AQLprofile/ROCProfiler. No direct SQTT/ATT kernel register-programming change is required.

Running thread tracing:
```bash
$ROCM_PATH/bin/rocprofv3 --att -d <out_dir> -o <prefix> -- python <script>.py
```

For detailed traces on gfx1151, prefer the triple-buffer path when available:
```bash
$ROCM_PATH/bin/rocprofv3 \
  --att \
  --att-triple-buffer \
  --att-buffer-size 0x100000 \
  --att-target-cu 1 \
  --att-simd-select 0x0 \
  --att-shader-engine-mask 0x1 \
  -d <out_dir> -o <prefix> -- python <script>.py
```

The current clean gfx1151 triple-buffer path was validated with one shader engine and reduced scope. FP16 `N=8192` detailed traces with target CU 1, SIMD 0, shader-engine mask `0x1`, and 1 MiB or 16 MiB ATT buffers produce decoded stats and `.att` files without `SQTT data out of bounds`, `Thread trace buffer full!`, GPU reset, or VM fault. The large `N=8192` trace can still report `Wave incomplete`, so treat per-wave timelines as partial when that warning appears.

- By default, it only traces the first dispatch of each kernel. If your script loops over the kernel multiple times, only the first instance is captured, so reduce `N_ITER` when possible.
- If the target kernel is not the first dispatch (e.g. random generation or copy kernels run first), inspect `ui_output_agent_*/code.json` to find the correct traced dispatch.
- `--kernel-include-regex <regex>` also applies to thread-trace data and is useful for excluding helper kernels from the decoded output.
- On gfx10/11, `--att-simd-select` is a SIMD ID (`0x0..0x3`), not a bitmask. `0x0` is a good starting choice on gfx1151.
- If a run exits successfully but the selected CU/SIMD saw no target wave, current local tooling warns `ATT decode produced no target CU/SIMD wave instruction records` and emits zero-hit disassembly for occupancy-reported kernels. Retry with a larger workload or a different `--att-target-cu`/`--att-simd-select` for instruction timing data.
- A safer reduced-scope single-buffer command is:
```bash
$ROCM_PATH/bin/rocprofv3 \
  --att \
  --att-buffer-size 0x1000000 \
  --att-target-cu 1 \
  --att-simd-select 0x0 \
  --att-shader-engine-mask 0x1 \
  -d <out_dir> -o <prefix> -- python <script>.py
```
- For unknown or unstable kernels on gfx1151, prefer no-detail ATT first:
```bash
$ROCM_PATH/bin/rocprofv3 \
  --att \
  --att-no-detail \
  --att-target-cu 1 \
  --att-simd-select 0x0 \
  --att-shader-engine-mask 0x1 \
  -d <out_dir> -o <prefix> -- python <script>.py
```
- Smaller `--att-buffer-size` values are valid (minimum 1 MB), but if you see `Thread trace buffer full!`, the trace is partial and you should increase the buffer or reduce trace scope/workload.
- If you use `--att-consecutive-kernels`, rocprofv3 switches to device-mode ATT. In that mode, `--att-serialize-all` is invalid and setup fails with error 19 (`INVALID_ARGUMENT`).
- For new kernels or if a full-size trace causes a reset, first try a smaller problem size (e.g. `N=2048` or `N=4096`) and the reduced-scope or triple-buffer command above before attempting `N=8192`.
- `--att-no-detail` is validated on this machine as a low-volume fallback. Detailed FP16 `N=8192` with triple-buffering is validated for artifact/stats generation on this machine with 1 MiB and 16 MiB ATT buffers, but `Wave incomplete` means some per-wave trace timelines can still be partial.
- `Wave incomplete` means the decoder saw a wave start but did not see its matching `WAVE_END` token before the captured trace ended. This is a partial-tail warning, not the same as `Data Lost` or `Thread trace buffer full!`.
- If you see `SQTT data out of bounds` on gfx1151 finalization, verify that you are using a build with the gfx115x write-pointer offset fix. Valid hardware `SQ_THREAD_TRACE_WPTR` offsets are multiplied by 32 and used directly when within buffer capacity.

Outputs:
- `stats_*.csv`: Aggregated latency, stall, and idle cycle counts for every instruction in the kernel.
- `*.att`: Raw SQTT binary trace data.
- In triple-buffer mode, multiple chunks for one dispatch append to the same `.att` file.
- `ui_output_agent_*/`: Directory containing per-wave JSON files (e.g. `se0_sm0_sl0_wv0.json`), `code.json`, and occupancy/timeline metadata.
- In `--att-no-detail` mode, `code.json` is expected to contain `"code": null` and `stats_*.csv` may contain only the header row. The useful outputs are `occupancy.json`, `wstates*.json`, `realtime.json`, and the wave-slot JSON files.

### Analyzing Thread Trace Data

For human, you can load the `.att` file or the `ui_output_agent_*/` directory into ROCprof Compute Viewer to visualize the overlap between memory operations (LDS/Global) and compute (WMMA) on a timeline.

For AI, you can write python scripts to analyze the `wave_*.json` and `code.json` pairs. You can plot the timeline using matplotlib, and use your visual ability to assess it.

Pitfalls:
- Single-Wave Perspective: A trace timeline only plots the execution of one specific wave on one specific SIMD. If you see a massive sync stall (e.g. `s_waitcnt vmcnt`), that single wave is indeed completely stalled.
- Macro-Level Hiding (Occupancy): Do not confuse a single-wave stall with global GPU starvation. For example, a single wave might spend 60% of its time stalled on global memory (`vmcnt`), but if occupancy is 8 waves per SIMD, the hardware scheduler seamlessly context-switches to the other 7 resident waves. This allows the physical Vector ALUs (Matrix Cores) to remain busy computing `v_wmma` instructions for other waves, hiding the latency globally and achieving high TFLOPS (e.g., ~75% utilization).
- Internal Bottleneck (The "FP Convert Tax"): To find the true internal kernel bottleneck, calculate the ratio of instructions executed within the active math loop phase (ignoring the global wait stalls). For instance, unpacking fp8 data on RDNA3.5 requires `v_perm_b32` (VALU), which cannot execute concurrently with `v_wmma`. If this unpacking takes 23% of the math pipeline time, your maximum possible WMMA utilization is hard-capped at 77%.
- SQTT Profiling Overhead: Thread tracing (`--att`) forces the hardware sequencer (SQ) to stall instruction issue when its internal trace token FIFO fills up, waiting for trace data to be written to VRAM (`SQ_STALL_EN` and `SPI_STALL_EN`). This stuttering loop artificially inflates the wall-clock cycles between instructions. Therefore, if you manually calculate WMMA utilization from trace data (e.g., `total_wmma_busy_cycles / total_trace_duration_cycles`), the result (e.g., ~30%) will be drastically lower than the true benchmarked utilization (e.g., ~75%) because the trace duration is heavily inflated by profiling stalls.
