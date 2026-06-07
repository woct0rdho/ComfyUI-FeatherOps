# TensileLite Unit Test Notes

These notes describe the local workflow for running TensileLite Python unit tests after changing `~/rocm-libraries/projects/hipblaslt/tensilelite`, and for rebuilding the in-tree `rocisa` Python extension when needed.

## Assumptions

- Use the current activated Python venv. Do not create a tox env unless you explicitly want tox isolation.
- `ROCM_PATH` is already set to the ROCm SDK root in the Python venv.
- Run commands from the TensileLite source root unless noted:

```bash
cd ~/rocm-libraries/projects/hipblaslt/tensilelite
```

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

## Run Targeted Unit Tests

Put the source checkout first on `PYTHONPATH` so tests import the edited TensileLite Python files:

```bash
cd ~/rocm-libraries/projects/hipblaslt/tensilelite
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Run targeted non-GPU tests relevant to the local patches, such as:

```bash
pytest \
  Tensile/Tests/unit/test_CustomSchedule.py \
  Tensile/Tests/unit/test_ValidatePack.py \
  Tensile/Tests/unit/test_validateParameterTypes.py
```

`Tensile/Tests/unit/test_MatrixInstructionConversion.py` may be slow (~2 minutes) at collecting tests.

The following tests are gated to gfx950:

```text
Tensile/Tests/unit/test_gr_lr_roundtrip.py
Tensile/Tests/unit/test_lraTileAssignment.py
Tensile/Tests/unit/test_storeD_roundtrip.py
Tensile/Tests/unit/test_TensileLibLogicToYaml.py
```

The following tests are gated to gfx1250:

```text
Tensile/Tests/unit/test_gl2_prefetch_offset.py
```

## Run `rocisa` Tests

After `rocisa` C++ changes, run at least the direct `rocisa` test tree:

```bash
pytest rocisa/test
```

## Common Tests With A Prebuilt Client

The Python unit tests do not need `tensilelite-client`. YAML/common tests do. Rebuild the client when client C++ changes or when validating generated kernels through the TensileLite common-test path:

```bash
~/rocm-libraries/build_tensilelite_client.sh
```

Then run a focused common-test set with the prebuilt client and current source checkout:

```bash
pytest \
  --prebuilt-client ~/rocm-libraries/build/tensilelite-client/tensilelite/client/tensilelite-client \
  --global-parameters CheckASMCodeSize=True \
  --gpu-targets gfx1151 \
  Tensile/Tests/common/gemm/gfx11/fp16_tn_gfx11.yaml
```

For routine local codegen edits, start with targeted unit tests plus direct TensileLite generation/validation of the HHH/HHS candidates described in `doc/tensile_fp16_nt_hhh.md` and `doc/tensile_fp16_nt_hhs.md`.
