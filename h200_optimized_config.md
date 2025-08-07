# 🔥 H200 SXM Configurazioni Ottimizzate per XStar Training

## ⚡ Setup Consigliato per H200 (139GB VRAM)

### 1. Ultra High Quality (30-40 min)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py train \
    --epochs 150 \
    --batch-size 24 \
    --image-size 1024 1024 \
    --experiment-name "h200_ultra_hq_v2"
```

### 2. Balanced Performance (20-25 min)  
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py train \
    --epochs 120 \
    --batch-size 32 \
    --image-size 896 896 \
    --experiment-name "h200_balanced"
```

### 3. Maximum Speed (15-20 min)
```bash  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py train \
    --epochs 100 \
    --batch-size 40 \
    --image-size 768 768 \
    --experiment-name "h200_speed"
```

### 4. Safe Mode (Garantito funzionante)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py train \
    --epochs 80 \
    --batch-size 16 \
    --image-size 640 640 \
    --experiment-name "h200_safe"
```

## 🎯 Calcoli Memoria H200

- H200 Total VRAM: 141GB HBM3
- Model Parameters: ~13.7M (0.5GB)  
- Optimizer State: ~1.5GB
- Available for Batches: ~135GB

### Memory Usage per Configuration:
- 1280x1280, BS48: ~140GB ❌ (Too close to limit)
- 1024x1024, BS24: ~85GB ✅ (Safe margin)  
- 896x896, BS32: ~75GB ✅ (Optimal balance)
- 768x768, BS40: ~60GB ✅ (Max throughput)

## 🚀 One-Liner Raccomandato (25 minuti):
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python main.py train --epochs 120 --batch-size 32 --image-size 896 896 --experiment-name "h200_optimal"
```
