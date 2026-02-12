# HIP vs torch.compile Match Log (gfx1151, N=8192)

## Snapshot
Goal:
- Beat `torch.compile` on `gfx1151` while keeping fused dataflow:
  - fp8 global load -> LDS
  - fp8->fp16 conversion inside GEMM path
  - avoid full-tensor global fp16 pre-cast

Accuracy gate:
- `relative L2 <= 0.01`
- `max abs <= 1.0`

Approximation policy:
- approximation in fp8->fp16 / GEMM internals is allowed if accuracy gate stays green
- denorm/NaN exact behavior may be relaxed if gate stays green

Scope:
- Target GPU: `gfx1151`
- Active mode: single fixed config `2,4,2,2,4,4`; autotune disabled for this phase
- `stages` fixed at `2` (do not re-open stage>2 search)
- Final performance indicator: unprofiled benchmark median GFLOPS from `benchmark_scaled_mm_hip.py` (`N=8192`)
- Profiling (`rocprofv3`) is diagnostics-only for bottleneck analysis and conflict attribution
- B-conversion policy for current cycle: no inline-asm conversion path

Torch steady reference (`TORCH_COMPILE_REFERENCE.md`):
- fp8->fp16 cast: `0.952 ms`
- GEMM: `40.186 ms`
- epilogue: `1.039 ms`
- total: `42.176 ms`

Current baseline (benchmark-first):
- benchmark (`N=8192`, unprofiled median): `34754.230 GFLOPS` (`rocprof_hip_8192_step29_remove_kprofile_mode/scaled_mm_hip_step29.csv`)
- profiled kernel time (diagnostic): `32.231 ms` median, `32.271 ms` avg (`rocprof_hip_8192_step29_remove_kprofile_mode/rocprof/hip8192_step29_results.db`)
- resources: `VGPR=192`, `SGPR=128`, `LDS=25088`, `scratch=0`, `wg=256`

## Non-Negotiable Run Protocol
1. Never run two benchmark/profile jobs at the same time. Before benchmark/profile, gate with:
   - `ps -eo pid,cmd | rg -n "benchmark_scaled_mm_hip.py|profile_scaled_mm_hip.py|rocprofv3" -S`
2. Per-step order:
   - `python test_scaled_mm_hip.py`
   - `python benchmark_scaled_mm_hip.py`
   - `rocprofv3 --kernel-trace --stats -d ... -o ... -- python -u profile_scaled_mm_hip.py -N 8192 --iters 20`
3. Revert failed steps via scoped `git diff` rollback. Skip test/benchmark/profile after revert (unless explicitly requested).
4. If a new baseline is kept, commit the kernel immediately.
5. After every experiment, update this file with findings, keep/reject, regression reason, next steps.
6. Do not repeat experiments already completed in this file unless there is a clearly new precondition (different schedule structure, different ISA path, or changed contract).
7. Continue autonomously to the next experiment. Do not stop and wait for the user's confirmation, unless blocked by unrecoverable error or the user explicitly interrupted.

## Phase Status
- Phase 1 (A/B load): closed
- Phase 2 (pipeline schedule): closed
- Phase 3 (asm in pipeline schedule): closed
- Phase 4 (C store and epilogue): closed

## Autotune Config Sweep (Latest Useful Facts)
Interface/tooling:
- `benchmark_scaled_mm_hip_configs.py` supports non-square shapes:
  - `--shapes "M,N,K;M,N,K;..."`
  - `--sizes` remains square shorthand

Compile-time constraint:
- `(2,4,2,2,2,4)` invalid for this kernel family:
  - fails `kShASize >= kCShuffleSize` static assert in `hip_kernel.cu`
  - reason: C-shuffle LDS reuse requires enough `repeat_m` with `warps_n=4`

Ranking summary (from config sweep):
- square / wide-N dominant shapes: best `(2,4,2,2,4,4)`
- tall-narrow-N shapes: `(2,4,2,2,4,2)` or `(4,2,2,2,2,4)` can be stronger
- consistently weak configs removed in prior sweep:
  - `(1,4,2,2,8,2)` and tested 256x64 variants `(2,2,2,2,8,2)`, `(4,1,2,2,4,4)`, `(4,2,2,2,4,2)`

## High-Signal Kept Wins
Schedule/A-path progression:
- Step24: removed coarse preload barrier -> `28779.687 GFLOPS`, `38.537 ms`
- Step33: split preload order `A then B` -> `30335.689 GFLOPS`, `37.744 ms`
- Step42: compile-time contig fastpath -> `30846.277 GFLOPS`, `35.653 ms`
- Step53: per-`u` `lgkmcnt(0)` cadence -> rerun band `30944~31398 GFLOPS`, `35.372~35.642 ms`
- Step65: selective `s_setprio` + Step53 cadence -> `32404.337 GFLOPS`, `33.984 ms`
- StepB12: A physical+inverse mapping + WSGR A-store ownership -> `34403.995 GFLOPS`

C/epilogue progression:
- StepC02 (interior/edge split): KEEP (neutral-to-slightly-positive after rerun)
- StepC04 (C-shuffle physical row mapping): KEEP -> `34746.607 GFLOPS`
- StepC09 (remove `kProfileMode`/split profiling path): KEEP -> `34754.230 GFLOPS` (current baseline)

## Important Rejections (Do Not Re-Run Without New Preconditions)
B-path structural attempts:
- StepB13 WSGRB overlap owners: REJECT (`25952.205 GFLOPS`, bottlenecked load issue)
- StepB14 WSGRB split owners: REJECT (`30853.477 GFLOPS`, lower B-store issue throughput)
- StepB15 B physical/inverse map: REJECT (`33598.657 GFLOPS`, overhead > gain)
- StepB17 packed-u8 pair conversion: REJECT (`33618.835 GFLOPS`, benchmark KPI down)
- StepB18 asm-u8 conversion: REJECT (`29146.327 GFLOPS`, VALU/scheduling cost)

C/epilogue micro-variants:
- StepC03 compile-time `has_scale/has_bias`: REJECT (`34270.151 GFLOPS`)
- StepC05 bias `half2` pairing: REJECT (`34472.830 GFLOPS`)
- StepC06 `kCPad=0`: REJECT (`34397.170 GFLOPS`)
- StepC07 `kCPad=16`: REJECT (`34127.070 GFLOPS`)
- `kCPad` conclusion under fixed config: `8` is best among `{0,8,16}`

WSGRC feasibility:
- StepC08: closed as infeasible/negative for current ownership model
- reason: accumulators are wave-private; cross-wave store ownership needs cross-wave exchange and harms coalesced store mapping

## Bank-Conflict Facts (Key)
Torch tuned GEMM:
- raw counter evidence of zero LDS conflict:
  - `SQC_LDS_BANK_CONFLICT=0`
  - `SQC_LDS_IDX_ACTIVE=21,609,054,208`
- artifact: `rocprof_bankconf_torch_8192_raw/rocprof/torch_raw_results.db`

HIP (instrumented branch facts from prior round):
- A path: dominant residual conflicts on LDS write side (A-read-only can be `0.0` in isolation)
- B path: write-dominant conflict pattern (`mode9` high conflict, `mode10` near-zero)
- these were diagnostic-only and were removed from production kernel path in StepC09

## No-Repeat Map
Do not retry without a clearly different precondition:
- WSGRB owner-wave variants from B13/B14
- B physical/inverse row mapping from B15 (same mapping form)
- packed-u8 pair conversion from B17 (same formulation)
- asm-u8 conversion from B18
- prior A local-read parity/phased variants (A03/A04/A07/A08)
- one-buffer interleave/refill patterns (Step80/87/94 class)
- `s_waitcnt_depctr` transition tweak (Step108)

## Artifact Index
Torch:
- `rocprof_torch_8192_steady/torch8192_steady_results.db`
- `rocprof_bankconf_torch_8192_raw/rocprof/torch_raw_results.db`

HIP baseline:
- `rocprof_hip_8192_step29_remove_kprofile_mode/rocprof/hip8192_step29_results.db`
