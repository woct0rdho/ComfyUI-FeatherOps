# TensileLite Unit Test Notes

These notes describe the local workflow for running the tests that cover our carried changes to `~/rocm-libraries/projects/hipblaslt/tensilelite`, and for rebuilding the in-tree `rocisa` Python extension when needed.

Our branch carries these changes (see the `develop` commit log), and the test list below is scoped to them:
- local-read pack allocation, local-read element width, WMMA v1 d16 local reads, bounded local-read VGPR buffers, and the VALU-first WMMA VGPR layout.
- guarded static WGM8 workgroup mapping.
- StreamK with `UseE` and the StreamK partial-workspace store tests (the code fix itself landed upstream in PR `#14338` as an `offsetCode` split, so our commit keeps only the GTest and the characterization test).
- the gfx1151 HHS/BBS GridBased tuning logic, including backfilled `SolutionUID` fields.
- `MaxLgkmcnt=63` for gfx115x (in `shared/tensile`).
- AK16 mixed float8 (`TensorALayoutA`) support.
- local build scripts.

Upstream also reworked the subsystems these changes sit in: the package rename to `tensilelite`, the persistent execution policy / `isPersistent`, `ReuseAcrossPersistent`, the `UseE` paths, GL2 prefetch, StreamK 2D clusters and their scheduling ABI2, and the complex-B local-read wait tuning.

The same period brought MX 128 scaling blocks (`MXBlockFreeA`/`MXBlockFreeB`, which changed the MX pack-index condition our local-read commit merges), paired LDS buffers for tight segment interleave, the independent TensileLite packaging, and the `SolutionUID` and `stochasticRounding` migrations.

The upstream tests that cover those subsystems are included below.

## Assumptions

- Use the current activated Python venv. Do not create a tox env. tox does not work on this machine.
- `ROCM_PATH` is already set to the ROCm SDK root in the Python venv.
- Run commands from the TensileLite source root unless noted:

```bash
cd ~/rocm-libraries/projects/hipblaslt/tensilelite
```

- Upstream moved the Python package from `Tensile/` to `tensilelite/` (`refactor(tensilelite): move package to lowercase namespace`). Tests now live under `tensilelite/Tests/unit` and `tensilelite/Tests/common`. The remaining `Tensile/` directory is only a compatibility alias, so `from Tensile. ... import ...` still works in tests and scripts.

## Rebuild `rocisa`

Rebuild `rocisa` when any of these change:
- `tensilelite/rocisa/rocisa/**/*.cpp`, `*.hpp`, `*.h`, `*.def`, or `*.inc`.
- Shared C++ dependencies used by the extension, especially `~/rocm-libraries/shared/stinkytofu`.
- `tensilelite/rocisa/CMakeLists.txt` or `tensilelite/rocisa/pyproject.toml`.
- The import-time stale-binding check says `_rocisa.so` is older than sources.

Use ROCm clang explicitly. Without `CC`/`CXX`, scikit-build may configure with system `g++`, which can fail on newer GCC warnings.

```bash
cd ~/rocm-libraries/projects/hipblaslt/tensilelite

CC="$ROCM_PATH/lib/llvm/bin/amdclang" \
CXX="$ROCM_PATH/lib/llvm/bin/amdclang++" \
CMAKE_ARGS='-DROCISA_INCLUDE_BUILD_INFO=ON' \
pip install -U --no-build-isolation -e ./rocisa
```

Check that the binding matches the sources before running anything else:

```bash
python -m pytest -q -p no:warnings tensilelite/Tests/unit/test_rocisa_install_status.py
```

## Run Targeted Unit Tests

Put the source checkout first on `PYTHONPATH` so tests import the edited TensileLite Python files:

```bash
cd ~/rocm-libraries/projects/hipblaslt/tensilelite
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

These unit tests run as-is. The common tests below additionally need `-p no:transformersci_otel` and a specific `LD_LIBRARY_PATH` order. The reason is documented in that section.

Tests added by our commits:

```bash
python -m pytest -q \
  tensilelite/Tests/unit/test_local_read_pack_allocation.py \
  tensilelite/Tests/unit/test_valu_first_vgpr_layout.py \
  tensilelite/Tests/unit/test_static_wgm.py \
  tensilelite/Tests/unit/characterization/_codegen/test_streamk_group_load_store_char.py
```

Adjacent tests for the code these commits touch (local-read/LDS packing, WMMA VGPR budget, resource limits):

```bash
python -m pytest -q \
  tensilelite/Tests/unit/test_lds_layout.py \
  tensilelite/Tests/unit/test_segment_interleave_policy.py \
  tensilelite/Tests/unit/test_MXBlockTile.py \
  tensilelite/Tests/unit/test_mx_free_phase_validation.py \
  tensilelite/Tests/unit/test_mx_scale_row_pitch.py \
  tensilelite/Tests/unit/test_mx_tile_tdm.py \
  tensilelite/Tests/unit/characterization/_codegen/test_mx_block128x128_lra_char.py \
  tensilelite/Tests/unit/characterization/LocalRead/test_r3_localread_char.py \
  tensilelite/Tests/unit/test_initc_iter_wmma.py \
  tensilelite/Tests/unit/test_occupancy.py \
  tensilelite/Tests/unit/test_occupancy_buildtime.py \
  tensilelite/Tests/unit/test_vgpr_overflow_force_generate.py \
  tensilelite/Tests/unit/test_register_overflow_deferral.py \
  tensilelite/Tests/unit/test_preloop_local_read_drain.py \
  tensilelite/Tests/unit/test_ValidatePack.py \
  tensilelite/Tests/unit/test_wgmxcc_kernel_name.py
```

`test_vgpr_overflow_force_generate.py` verifies that VGPR overflow warns and reaches `checkResources()` instead of failing during `_initKernel`, while AGPR overflow remains fatal. That distinction allows `ForceGenerateKernel=1` to preserve rejected kernel source for debugging.

`test_register_overflow_deferral.py` covers the GPU-free half of the same register-limit path: `judgeRegisterBudget` records a count triple, `deferRegisterOverflowForAllocation` reopens the verdict under force apply (`StinkyTofuRegisterAllocation` 3), and `resolveDeferredRegisterOverflow` settles it on the counts the emitted descriptor declares.

Upstream tests for the subsystems our commits modify.

`test_ReuseAcrossPersistent.py` matters most: the `ReuseAcrossPersistent` alignment and our VALU-first allocation now share the same code in `KernelWriter.py`, so this test is what guards that merge.

The persistent-policy tests cover `ExecutionPolicy.isPersistent()`, which our `UseE` change now relies on:

```bash
python -m pytest -q \
  tensilelite/Tests/unit/test_ReuseAcrossPersistent.py \
  tensilelite/Tests/unit/test_execution_policy.py \
  tensilelite/Tests/unit/test_execution_policy_ingestion.py \
  tensilelite/Tests/unit/test_data_parallel_static.py \
  tensilelite/Tests/unit/test_persistent_loop_reuse.py \
  tensilelite/Tests/unit/test_persistent_argument_layout.py \
  tensilelite/Tests/unit/test_persistent_assignment_execution.py \
  tensilelite/Tests/unit/test_persistent_config_generation.py \
  tensilelite/Tests/unit/test_register_kernel_args.py
```

`test_complex_b_read_wait.py` guards the complex-B `secondWaitIndex` wait that upstream added in the same branch as our AK16 `convertAfterDSB8` condition in `KernelWriter._makeSubIterSchedule`. The other tests here guard the codegen the rebase merged:

```bash
python -m pytest -q \
  tensilelite/Tests/unit/characterization/_codegen/test_complex_b_read_wait.py \
  tensilelite/Tests/unit/characterization/_codegen/test_streamk_cluster_completion.py \
  tensilelite/Tests/unit/characterization/_codegen/test_lsu_alpha_before_loadc.py \
  tensilelite/Tests/unit/test_cluster_entry_handoff_codegen.py \
  tensilelite/Tests/unit/test_streamk_occupancy_probe.py \
  tensilelite/Tests/unit/test_stochastic_rounding_library.py
```

`test_streamk_cluster_completion.py` and `test_cluster_entry_handoff_codegen.py` cover the 2D StreamK cluster and flag codegen.

`test_lsu_alpha_before_loadc.py` covers the LocalSplitU alpha ordering path in the store code that our `UseE` change edits.

`test_stochastic_rounding_library.py` exercises the `ProblemType` class whose `StateKeys` list our AK16 commit extends with `tensorALayoutA`, next to the new `stochasticRounding` field.

`test_streamk_occupancy_probe.py` covers the hybrid StreamK occupancy probe.

`test_data_parallel_static.py` is upstream's rename of the `test_streamk_dponly_sgpr_reduction.py` that earlier revisions of this document listed.

StreamK scheduler and rejection tests around the `UseE` and partial-workspace changes:

```bash
python -m pytest -q \
  tensilelite/Tests/unit/test_prefetchgl2_streamk_guard.py \
  tensilelite/Tests/unit/test_subtile_gl2_prefetch.py \
  tensilelite/Tests/unit/test_gl2_prefetch_offset.py \
  tensilelite/Tests/unit/test_streamk_force_dp_only.py \
  tensilelite/Tests/unit/test_streamk_uso_gate.py \
  tensilelite/Tests/unit/test_streamk_uso_mapping.py \
  tensilelite/Tests/unit/test_halfplr_streamk_rejects.py \
  tensilelite/Tests/unit/test_PrefetchAcrossPersistent.py \
  tensilelite/Tests/unit/test_streamk5_hybrid.py \
  tensilelite/Tests/unit/test_streamk_queue_index.py \
  tensilelite/Tests/unit/test_streamk_work_stealing.py
```

- `test_gl2_prefetch_offset.py` is the direct production address-generation verifier. Its gfx1250-gated cases skip on gfx1151 (`GL2 prefetch tests require gfx1250`). The CPU-side cases still run and pass.
- `test_lds_layout.py` checks the reported LDS addresses, the buffer copies and the segment-boundary clipping that the local-read changes rely on.
- `test_segment_interleave_policy.py` exercises the paired-placement policy for tight LDS segment interleave through gfx1250 `Solution` derivation.
- `test_MXBlockTile.py` (with the `test_mx_*` files and `test_mx_block128x128_lra_char.py`) covers the `MXBlock{A,B}`/`MXBlockFree{A,B}` tile spelling and scale-row mapping. Run these with the local-read set: our pack-index merge keeps upstream's `mxUnitA == 1` condition next to `localReadNeedsPack`.
- `test_prefetchgl2_streamk_guard.py` validates the DP-first StreamK PrefetchGL2 guard with both `StreamKForceDPOnly` modes. It uses gfx1250 assembler/capability data but does not need a gfx1250 GPU. It skips if the installed `amdclang++` cannot describe gfx1250.
- `test_subtile_gl2_prefetch.py` is a Python-only scheduler test and does not require a GPU.

Upstream's PR `#14338` fixed the same partial-store bug by routing the `tmpS01` offset updates into the same deferred module as the stores, so each store's offset update travels with it instead of being emitted earlier.

The GTest and `test_streamk_group_load_store_char.py` above assert the resulting invariant (an offset update between consecutive workspace stores) rather than the implementation, so they pass against either form.

Upstream also ships `tensilelite/Tests/common/streamk/gfx1250/core/sk_group_load_store_partials.yaml`, which needs gfx1250 hardware. Run it through the build-only entry point described below when no gfx1250 device is available.

Upstream now provides the real `SAddU64`/`VAddNCU64`/`GlobalPrefetchB8` rocisa instructions, so our old local compatibility shim for `Components/GL2Prefetch.py` was dropped. The GL2 tests above guard the production address generation instead.

The AK16 (`TensorALayoutA`) support has no unit test of its own. It is exercised through the client's `--tensor-a-layout-a` option and validated at runtime by the verifier and the hipBLASLt GTests below, plus the tuned gfx1151 logic that uses it.

## SolutionUID Tests

Upstream backfilled every shipped logic solution with a `SolutionUID` (`chore(hipblaslt): backfill SolutionUID in logic YAML`, PR `#13148`).

Our four tuned gfx1151 files were regenerated before that migration, so the rebase added the missing UIDs (125 per file for the 1135-shape grid commit, 180 per file for the StreamK commit: 1220 new UIDs, placed after `SolutionNameMin` the way upstream places them in legacy list-format logic).

Run the maintenance and repo-wide gate tests after touching any logic YAML:

```bash
python -m pytest -q \
  tensilelite/Tests/unit/test_solution_uid_uniqueness.py \
  tensilelite/Tests/unit/test_solution_uid_base62.py \
  tensilelite/Tests/unit/test_solution_uid_concurrency.py \
  tensilelite/Tests/unit/test_tensile_generate_uid.py \
  tensilelite/Tests/unit/test_LibraryIO.py \
  tensilelite/Tests/unit/test_TensileMergeLibrary.py \
  tensilelite/Tests/unit/test_TensileLibLogicToYaml.py
```

`test_solution_uid_uniqueness.py` scans the whole shipped corpus (`projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/**/*.yaml`, override with `HIPBLASLT_LOGIC_ROOT`) and skips when that tree is absent.

A missing `SolutionUID` is tolerated unless `HIPBLASLT_REQUIRE_SOLUTION_UID` is set, but the shipped corpus convention is that every solution has one.

The format and the single-solution regeneration command are documented in `projects/hipblaslt/docs/solution-uid.md`.

## Characterization Suites

The characterization tests re-generate kernels for curated logic files and compare digests or snapshots. They are the regression net for the shared emitter (KernelWriter / KernelWriterAssembly / `Components/*`) that our local-read, VALU-first, StreamK and AK16 changes touch, across architectures:

```bash
python -m pytest -q tensilelite/Tests/unit/characterization/_codegen/
```

```bash
python -m pytest -q \
  tensilelite/Tests/unit/characterization/Activation \
  tensilelite/Tests/unit/characterization/Architectures \
  tensilelite/Tests/unit/characterization/Contractions \
  tensilelite/Tests/unit/characterization/LibraryIO \
  tensilelite/Tests/unit/characterization/LibraryLogic \
  tensilelite/Tests/unit/characterization/LocalRead \
  tensilelite/Tests/unit/characterization/ProblemType \
  tensilelite/Tests/unit/characterization/SolutionLibrary \
  tensilelite/Tests/unit/characterization/SolutionStructsUtils \
  tensilelite/Tests/unit/characterization/TensileLibLogicToYaml
```

(Run the second command without the directory list to cover all 64 subdirectories. The shortened list is the subset nearest to our changes.)

The `ProblemType` goldens pin every default in `_defaultProblemType` and the state-key count, so adding a default changes them. Our AK16 commit adds `TensorALayoutA`, which updates exactly three files:
- `ProblemType/__snapshots__/test_helpers_char.ambr` (the parameter-type roster)
- `ProblemType/__snapshots__/test_problemtype_char.ambr` (the state dumps and the key count)
- `SolutionDerivationSweep/__snapshots__/test_setcover_reject_other_char.ambr` (only because the fp8 type-gate message also names bfloat8)

Regenerate only those nodes with `--snapshot-update` and keep the result in the commit that changes the behavior. The blanket-regeneration guard trips above three changed `.ambr` files, so this set needs no `Bulk-Snapshot-Update` ADR. Re-run the same nodes twice without `--snapshot-update` and confirm the files are byte-identical.

## `shared/tensile` Tests

`MaxLgkmcnt` is a `shared/tensile` capability, and our commit also added its probe test there:

```bash
cd ~/rocm-libraries/shared/tensile
PYTHONPATH="$PWD" python -m pytest -q -p no:warnings Tensile/Tests/unit/test_Common.py
```

## Run `rocisa` Tests

After `rocisa` C++ changes, run at least the direct `rocisa` test tree:

```bash
python -m pytest -q rocisa/test
```

The tree now also covers the rocisa instruction and caps changes that landed upstream: `test_mfma.py` (f8f6f4 MFMA format selectors), `test_rocisa_backend_selection.py` (backend and ISA selection), and `test_asm_caps_closed_stdin.py` (POSIX `run()` with a closed stdin).

## Common Tests With A Prebuilt Client

The Python unit tests do not need `tensilelite-client`. The YAML/common tests do. Rebuild the client whenever the client C++ or the generated library format changes:

```bash
~/rocm-libraries/build_tensilelite_client.sh
```

`HIPBLASLT_ENABLE_YAML` selects the host library backend instead of adding to it. The hipBLASLt superbuild forwards it to `TENSILELITE_ENABLE_YAML` (`projects/hipblaslt/CMakeLists.txt`):
- `build_tensilelite_client.sh` passes `-DHIPBLASLT_ENABLE_YAML=ON`, so the client's `build/tensilelite-client/tensilelite/libtensilelite-host.so.1` can read the YAML libraries that the Python driver writes by default, but it does not export `TensileLite::LoadLibraryMapping`.
- `build_hipblaslt.sh` leaves YAML off, so the installed `$ROCM_PATH/lib/libtensilelite-host.so.1.0` is msgpack-only: it exports `LoadLibraryMapping` (needed by the installed `libhipblaslt`) but cannot read YAML libraries.

Upstream also rejects the client, tests and autobuild targets in a standalone TensileLite configure (`TENSILELITE_STANDALONE`, when `tensilelite/` is the top-level CMake project) with `The TensileLite client, tests, and autobuild scripts require the hipBLASLt build.` `build_tensilelite_client.sh` configures `projects/hipblaslt`, so it stays on the supported path.

Consequences for running the common tests:
- Put the client's build directory first on `LD_LIBRARY_PATH`, otherwise the installed msgpack-only copy wins and every hardware config fails immediately with `Failed to load solution library`.
- Disable the `transformersci_otel` pytest plugin with `-p no:transformersci_otel`: it imports `torch`, which preloads the installed `libhipblaslt` that requires `LoadLibraryMapping` from the host library.

Common YAMLs are parameters of `tensilelite/Tests/common/test_config.py`. Naming a YAML directly does not collect a test. Select the exact parameterized node ID:

```bash
cd ~/rocm-libraries/projects/hipblaslt/tensilelite
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/rocm-libraries/build/tensilelite-client/tensilelite:$ROCM_PATH/lib:$ROCM_PATH/llvm/lib/x86_64-unknown-linux-gnu:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}"
CLIENT="$HOME/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client"

python -m pytest -q -p no:transformersci_otel \
  --prebuilt-client "$CLIENT" \
  --global-parameters CheckASMCodeSize=True \
  --gpu-targets gfx1151 \
  'tensilelite/Tests/common/test_config.py::test_config[tensilelite/Tests/common/gemm/gfx11/fp16_tn_gfx11.yaml]'
```

The upstream gfx11 pre-swizzle implementation from PR `#10733` has its own common test (separate from our AK16 `TensorALayoutA` byte-record layout):

```bash
python -m pytest -q -p no:transformersci_otel \
  --prebuilt-client "$CLIENT" \
  --global-parameters CheckASMCodeSize=True \
  --gpu-targets gfx1151 \
  'tensilelite/Tests/common/test_config.py::test_config[tensilelite/Tests/common/gemm/swizzle_gfx11.yaml]'
```

For the generalized/sparse GL2 path, use the split-CI build-only entry point when gfx1250 hardware is unavailable:

```bash
artifact_dir="$(mktemp -d /tmp/tensilelite-sparse-gl2.XXXXXX)"
python -m pytest -q -p no:transformersci_otel \
  --build-only \
  --artifact-dir "$artifact_dir" \
  --prebuilt-client "$CLIENT" \
  --global-parameters CheckASMCodeSize=True \
  --gpu-targets gfx1250 \
  'tensilelite/Tests/common/test_config_build.py::test_config_build[tensilelite/Tests/common/sparse/gfx1250/spmm_tdm_gl2prefetch.yaml]'
```

This compiles and packages gfx1250 code-generation coverage for sparse metadata plus GL2 prefetch without launching it on the local GPU. It does not replace the real-gfx1250 address verifier or the corresponding `test_config_run.py` artifact-consumption phase on gfx1250 hardware.

## hipBLASLt Bench and Tests After A Rebuild

These use the installed msgpack-only host library, so the order is the opposite of the common tests: keep `$ROCM_PATH/lib` first (the default `LD_LIBRARY_PATH` hinted by the build scripts) and do not put the client build directory in front.

```bash
cd ~/rocm-libraries/build/hipblaslt-bench/clients
HIPBLASLT_TENSILE_LIBPATH="$ROCM_PATH/lib/hipblaslt/library/gfx1151" \
LD_LIBRARY_PATH="$ROCM_PATH/llvm/lib/x86_64-unknown-linux-gnu:$ROCM_PATH/llvm/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}" \
./hipblaslt-test --gtest_filter='*quick*:*smoke*' --gtest_brief=1
```

`build_hipblaslt.sh` installs `libhipblaslt.so.1.5` (version `100500`) and makes `libtensilelite-host.so` a separately versioned runtime package, so the script also syncs the `libtensilelite-host.so` entries into `_rocm_sdk_libraries/lib`.

Keeping those links pointing at `$ROCM_PATH/lib` matters: a stale real file there silently shadows the freshly installed host library, which then pairs old host-side solution handling with the new device libraries.

Our StreamK partial-workspace commit added a targeted case. Run it explicitly after touching StreamK stores:

```bash
./hipblaslt-test --gtest_filter='*streamk_group_load_store*' --gtest_brief=1
```

Rebuild those clients first with `TARGET=hipblaslt-bench ./build_hipblaslt_bench.sh` and `TARGET=hipblaslt-test ./build_hipblaslt_bench.sh`. See `~/evotensile/README.md` for the full rebuild and verification sequence.

## Regenerating Checked-In Logic

After regenerating the checked-in GridBased logic for a target (see `~/evotensile/README.md`), confirm that no solution pins `InternalSupportParams.KernArgsVersion`:

```bash
rg -l 'KernArgsVersion' \
  ~/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/
```

TensileLite binds that key to the generator version, not to a tuning result. The upstream `pkareorder` change moved the default from `2` to `3` and reordered the host-side packed kernel arguments, and TensileLite drops the key when it writes logic so that logic files follow `defaultInternalSupportParams`.

A logic file that still pins the old value generates kernels for the old kernel-argument layout while the rebuilt host library packs the new one.

Kernel generation, the Python unit suites, and the common tests all succeed. The mismatch appears only when a kernel runs, as `an illegal memory access was encountered` (700) with `HSA_STATUS_ERROR_MEMORY_FAULT` and a gfxhub page fault in `dmesg`.

A stale pin therefore has to be caught by the runtime gates, not by the Python suites: run the EvoTensile verifier and the `quick`/`smoke` GTest filter on real hardware after any logic regeneration before trusting a rebuilt library.

Regenerated solutions also need `SolutionUID` fields to match the shipped corpus convention. See the SolutionUID section above and `projects/hipblaslt/docs/solution-uid.md`.
