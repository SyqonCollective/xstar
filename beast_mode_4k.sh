#!/bin/bash
# 🔥 BEAST MODE 4K - Velocità massima per immagini 4K
# Ottimizzato per 2x A40 (96GB VRAM)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:1024
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "🔥 BEAST MODE 4K - Maximum Speed Setup"
echo "======================================"
echo "🎯 Target: 4K images processing"
echo "🚀 Hardware: 2x A40 (96GB VRAM)"
echo ""

# 🔥 CONFIGURAZIONE BEAST per 4K (parametri completi)
python main.py train \
  --input-dir "Starlossno-1/starmodel dataset/input" \
  --starless-dir "Starlossno-1/starmodel dataset/starless" \
  --epochs 40 \
  --batch-size 6 \
  --tile-size 512 512 \
  --overlap 0.1 \
  --num-workers 8 \
  --experiment-name "beast_4k_training_fp16_accum" \
  --mixed-precision fp16 \
  --grad-accum 4 \
  --prefetch-factor 2 \
  --no-persistent-workers \
  --pin-memory \
  --learning-rate 0.002 \
  --weight-decay 1e-4 \
  --dropout 0.15 \
  --augmentation-factor 6 \
  --validation-split 0.15 \
  --early-stopping-patience 8 \
  --save-every 5 \
  --compile-model

echo ""
echo "🔥 Beast mode training completed!"
