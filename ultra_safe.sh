#!/bin/bash
# 🐌 ULTRA SAFE MODE - Per quando tutto va storto

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

echo "🐌 ULTRA SAFE MODE - Slow but sure"
echo "=================================="

python main.py train \
  --input-dir "Starlossno-1/starmodel dataset/input" \
  --starless-dir "Starlossno-1/starmodel dataset/starless" \
  --epochs 50 \
  --batch-size 8 \
  --tile-size 224 224 \
  --overlap 0.2 \
  --num-workers 8 \
  --learning-rate 0.0005 \
  --weight-decay 1e-5 \
  --dropout 0.1 \
  --augmentation-factor 4 \
  --validation-split 0.15 \
  --early-stopping-patience 15 \
  --experiment-name "ultra_safe_training" \
  --gradient-accumulation-steps 4 \
  --save-every 10
