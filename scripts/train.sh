#!/usr/bin/env bash

RUN_DIR="xxx/RCDFNet-main/"
GPU_ID=2
PYTHON_BIN="xxx/.conda/envs/RCDFNet3/bin/python"

cd "$RUN_DIR"

# Train RCDFNet on VoD dataset
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$RUN_DIR" \
"$PYTHON_BIN" tools/train.py \
    --config configs/RCDFNet/RCDFNet_VoD.py \
    --work-dir work_dirs/RCDFNet_VoD \
    --seed 0 \
    --deterministic

# Train RCDFNet on TJ4DRadSet dataset
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$RUN_DIR" \
"$PYTHON_BIN" tools/train.py \
    --config configs/RCDFNet/RCDFNet_TJ4D.py \
    --work-dir work_dirs/RCDFNet_TJ4D \
    --seed 0 \
    --deterministic