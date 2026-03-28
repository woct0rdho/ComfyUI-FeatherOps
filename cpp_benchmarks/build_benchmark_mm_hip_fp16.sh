#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BIN="$BUILD_DIR/benchmark_mm_hip_fp16"
KERNEL_SRC="$SCRIPT_DIR/../kernel/hip/hip_kernel_fp16.cu"
BENCH_SRC="$SCRIPT_DIR/benchmark_mm_hip_fp16.cpp"

CLANGXX="$ROCM_PATH/lib/llvm/bin/clang++"

mkdir -p "$BUILD_DIR"

"$CLANGXX" \
  -x hip \
  --offload-arch=gfx1151 \
  --rocm-path="$ROCM_PATH" \
  -I"$ROCM_PATH/include" \
  -L"$ROCM_PATH/lib" \
  -D__HIP_PLATFORM_AMD__ \
  -DNO_PYTORCH \
  -O3 \
  -std=c++17 \
  "$KERNEL_SRC" \
  "$BENCH_SRC" \
  -lamdhip64 \
  -o "$BIN"

echo "Built $BIN"
