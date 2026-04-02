#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

build_ext_in_dir() {
  local dir="$1"
  local required="${2:-1}"
  local setup_py="$ROOT_DIR/$dir/setup.py"

  if [[ ! -f "$setup_py" ]]; then
    echo "[SKIP] $dir/setup.py not found"
    return 0
  fi

  echo "[BUILD] $dir"
  if ! (
    cd "$ROOT_DIR/$dir"
    python setup.py build_ext --inplace
  ); then
    if [[ "$required" == "1" ]]; then
      echo "[ERROR] Failed building required op: $dir"
      return 1
    fi
    echo "[WARN] Failed building optional op: $dir"
  fi
}

# Core RCDFNet ops used by training/testing entrypoints.
build_ext_in_dir "rcdfnet/ops/iou3d" 1
build_ext_in_dir "rcdfnet/ops/voxel" 1
build_ext_in_dir "rcdfnet/ops/spconv" 1
build_ext_in_dir "rcdfnet/ops/bev_pool_v2" 1

echo "[OK] build_ops finished"
