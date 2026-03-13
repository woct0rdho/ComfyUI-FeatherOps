#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/benchmark_mm_hipblaslt_fp16"

CLANGXX="$ROCM_PATH/lib/llvm/bin/clang++"

"$CLANGXX" \
  -x hip \
  --offload-arch=gfx1151 \
  --rocm-path="$ROCM_PATH" \
  -I"$ROCM_PATH/include" \
  -L"$ROCM_PATH/lib" \
  -D__HIP_PLATFORM_AMD__ \
  -O3 \
  -std=c++17 \
  "$SCRIPT_DIR/benchmark_mm_hipblaslt_fp16.cpp" \
  -lhipblaslt \
  -lamdhip64 \
  -o "$BIN"

echo "Built $BIN"
