#!/bin/bash
# 🛢️ SMOOTH AS OIL - Configurazione ultra-ottimizzata

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

echo "🛢️ SMOOTH AS OIL Training Setup"
echo "================================"

python main.py train \
  --input-dir "Starlossno-1/starmodel dataset/input" \
  --starless-dir "Starlossno-1/starmodel dataset/starless" \
  --epochs 70 \
  --batch-size 16 \
  --tile-size 256 256 \
  --overlap 0.15 \
  --num-workers 12 \
  --learning-rate 0.001 \
  --weight-decay 1e-4 \
  --dropout 0.2 \
  --augmentation-factor 8 \
  --validation-split 0.2 \
  --early-stopping-patience 10 \
  --experiment-name "smooth_oil_training" \
  --mixed-precision \
  --gradient-accumulation-steps 2 \
  --save-every 5
