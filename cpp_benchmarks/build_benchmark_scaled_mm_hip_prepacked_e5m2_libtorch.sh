#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/benchmark_scaled_mm_hip_prepacked_e5m2_libtorch"
KERNEL_SRC="$SCRIPT_DIR/../kernel/hip/hip_kernel_prepacked.cu"
BENCH_SRC="$SCRIPT_DIR/benchmark_scaled_mm_hip_prepacked_e5m2_libtorch.cpp"

export PYTHONPATH="$SCRIPT_DIR"
PYTHON_CMD=${PYTHON_CMD:-python3}
TORCH_INCLUDES=$($PYTHON_CMD -c "import torch; import torch.utils.cpp_extension as cpp_ext; print(' '.join(f'-I{p}' for p in cpp_ext.include_paths()))")
TORCH_LIB_PATHS=$($PYTHON_CMD -c "import torch; import torch.utils.cpp_extension as cpp_ext; print(' '.join(f'-L{p} -Wl,-rpath,{p}' for p in cpp_ext.library_paths()))")
TORCH_LIBS="-ltorch -ltorch_cpu -ltorch_hip -lc10 -lc10_hip"

CLANGXX="$ROCM_PATH/lib/llvm/bin/clang++"

"$CLANGXX" \
  -x hip \
  --offload-arch=gfx1151 \
  --rocm-path="$ROCM_PATH" \
  -I"$ROCM_PATH/include" \
  -L"$ROCM_PATH/lib" \
  $TORCH_INCLUDES \
  $TORCH_LIB_PATHS \
  -D__HIP_PLATFORM_AMD__ \
  -DNO_PYTORCH \
  -O3 \
  -std=c++17 \
  "$KERNEL_SRC" \
  "$BENCH_SRC" \
  -lamdhip64 \
  $TORCH_LIBS \
  -o "$BIN"

echo "Built $BIN"
