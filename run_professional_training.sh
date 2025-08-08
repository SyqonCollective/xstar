#!/bin/bash
# Training PROFESSIONALE 70 epoche - 2x A40 OPTIMIZED
# MASSIMA AUGMENTATION + ANTI-OVERFITTING

echo "🚀 PROFESSIONAL 70-EPOCH TRAINING - 2x A40"
echo "🔥 MAXIMUM AUGMENTATION + ANTI-OVERFITTING"
echo "=========================================="

# Verifica GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

# Test PyTorch GPU
python -c "
import torch
print(f'🔥 CUDA: {torch.cuda.is_available()}')
print(f'🔥 GPUs: {torch.cuda.device_count()}')
total_mem = sum(torch.cuda.get_device_properties(i).total_memory/1e9 for i in range(torch.cuda.device_count()))
print(f'🚀 Total VRAM: {total_mem:.1f}GB')
"

echo "🔧 Starting PROFESSIONAL training..."
echo "✅ 70 epochs"
echo "✅ 12x augmentation factor (anti-overfitting)"
echo "✅ Batch size 28 (optimized for dual A40)"
echo "✅ 25% validation split"
echo "✅ Dropout 0.3 + Weight decay 1e-3"

# TRAINING COMMAND OTTIMIZZATO
python main.py \
  --input_dir "Starlossno-1/starmodel dataset/input" \
  --starless_dir "Starlossno-1/starmodel dataset/starless" \
  --epochs 70 \
  --batch_size 28 \
  --learning_rate 0.0001 \
  --experiment_name "professional_70epochs_max_aug" \
  --image_size 512 \
  --validation_split 0.25 \
  --augmentation_factor 12 \
  --dropout_rate 0.3 \
  --weight_decay 0.001 \
  --save_every 5 \
  --validate_every 2

echo "✅ PROFESSIONAL TRAINING COMPLETED!"
echo "📁 Check outputs/professional_70epochs_max_aug/ for results"
