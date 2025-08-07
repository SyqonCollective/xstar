# 🚀 XStar Training su RunPod - Guida Completa

## 📋 Setup Iniziale su RunPod

### 1. Crea Pod su RunPod
```bash
# Scegli template PyTorch con CUDA
# Raccomandato: RTX 3080/4080/4090 con almeno 12GB VRAM
# Template: PyTorch 2.0+ con Python 3.9+
```

### 2. Connetti al Pod
```bash
# Una volta avviato il pod, connettiti via Jupyter o SSH
# Apri il terminal integrato
```

## 🔽 Importazione Progetto da GitHub

### 3. Clona il Repository con Git LFS
```bash
# Installa Git LFS se non presente
sudo apt-get update && sudo apt-get install -y git-lfs

# Inizializza Git LFS
git lfs install

# Clona il repository con dataset completo
git clone https://github.com/SyqonCollective/xstar.git
cd xstar

# Verifica che il dataset sia stato scaricato
ls -la "Starlossno-1/starmodel dataset/"
find "Starlossno-1/" -name "*.jpg" | wc -l  # Dovrebbe essere 126
```

### 4. Setup Ambiente Python
```bash
# Aggiorna pip
python -m pip install --upgrade pip

# Installa PyTorch con CUDA (verifica versione CUDA del pod)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Installa dipendenze del progetto
pip install -r requirements.txt

# Verifica installazione
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 🧪 Test Pre-Training

### 5. Verifica Configurazione
```bash
# Test percorsi dataset
python test_dataset_paths.py

# Test GPU
python test_gpu.py

# Test quick dell'architettura
python -c "
from models.unet_starnet import UNet
import torch
model = UNet(in_channels=3, out_channels=3)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
if torch.cuda.is_available():
    model = model.cuda()
    x = torch.randn(1, 3, 512, 512).cuda()
    with torch.no_grad():
        y = model(x)
    print(f'Forward pass OK: {y.shape}')
"
```

## 🚀 Avvio Training

### 6. Training Rapido (Test)
```bash
# Training veloce per test (20 epochs)
python main.py train \
    --epochs 20 \
    --batch-size 16 \
    --image-size 512 512 \
    --experiment-name "runpod_test_run"
```

### 7. Training Completo (Produzione)
```bash
# Training completo ottimizzato per GPU
python main.py train \
    --epochs 200 \
    --batch-size 32 \
    --image-size 768 768 \
    --learning-rate 1e-4 \
    --experiment-name "runpod_production_v1"
```

### 8. Training con Parametri Avanzati
```bash
# Training professionale con tutti i parametri
python main.py train \
    --input-dir "Starlossno-1/starmodel dataset/input" \
    --starless-dir "Starlossno-1/starmodel dataset/starless" \
    --epochs 300 \
    --batch-size 24 \
    --image-size 768 768 \
    --learning-rate 1e-4 \
    --weight-decay 1e-5 \
    --experiment-name "runpod_final_model" \
    --save-every 25 \
    --validate-every 5
```

## 📊 Monitoraggio Training

### 9. TensorBoard (Opzionale)
```bash
# In un terminal separato
tensorboard --logdir outputs/runpod_production_v1/tensorboard --host 0.0.0.0 --port 6006

# Accedi tramite Public IP del pod sulla porta 6006
# Esempio: http://PUBLIC_IP:6006
```

### 10. Monitoraggio Real-time
```bash
# Monitora GPU usage
watch -n 1 nvidia-smi

# Monitora log training
tail -f outputs/runpod_production_v1/training.log

# Controlla spazio disco
df -h
```

## 💾 Backup e Download Risultati

### 11. Salvataggio Automatico
```bash
# I modelli vengono salvati automaticamente in:
# outputs/EXPERIMENT_NAME/
# - best_model.pth (migliore PSNR)
# - checkpoint_epoch_X.pth (checkpoint periodici)
# - training_metrics.json (metriche complete)
```

### 12. Download Risultati
```bash
# Comprimi risultati per download
tar -czf runpod_training_results.tar.gz outputs/

# Oppure usa il file manager di RunPod per scaricare:
# - outputs/EXPERIMENT_NAME/best_model.pth
# - outputs/EXPERIMENT_NAME/training_metrics.json
# - outputs/EXPERIMENT_NAME/sample_results/
```

## ⚡ Configurazioni Ottimizzate per GPU

### RTX 3080/4070 (12GB VRAM)
```bash
python main.py train \
    --epochs 200 \
    --batch-size 16 \
    --image-size 640 640 \
    --experiment-name "rtx3080_optimized"
```

### RTX 4080/4090 (16GB+ VRAM)
```bash
python main.py train \
    --epochs 300 \
    --batch-size 32 \
    --image-size 768 768 \
    --experiment-name "rtx4090_max_quality"
```

### GPU Memory Issues?
```bash
# Se hai problemi di memoria:
python main.py train \
    --epochs 150 \
    --batch-size 8 \
    --image-size 512 512 \
    --gradient-accumulation 4 \
    --experiment-name "low_memory_safe"
```

## 🎯 Risultati Attesi

### Performance Target su RunPod
- **Training Time**: 2-6 ore (dipende da GPU e parametri)
- **PSNR Target**: > 32 dB
- **SSIM Target**: > 0.95
- **Star Removal Rate**: > 0.92

### Output Files
```
outputs/EXPERIMENT_NAME/
├── best_model.pth          # Modello migliore
├── checkpoint_epoch_X.pth  # Checkpoint intermedi
├── training_metrics.json   # Metriche complete
├── tensorboard/            # Log TensorBoard
└── sample_results/         # Esempi risultati
```

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Riduci batch size
--batch-size 8

# Riduci dimensioni immagini
--image-size 512 512

# Abilita gradient checkpointing
--gradient-checkpointing
```

### Dataset Non Trovato
```bash
# Verifica Git LFS
git lfs pull

# Controlla percorsi
ls -la "Starlossno-1/starmodel dataset/"
```

### Training Lento
```bash
# Verifica GPU usage
nvidia-smi

# Aumenta num_workers
--num-workers 8
```

## 🎉 One-Liner Completo per RunPod

```bash
# Setup e avvio completo in un comando
sudo apt-get update && sudo apt-get install -y git-lfs && \
git lfs install && \
git clone https://github.com/SyqonCollective/xstar.git && \
cd xstar && \
python -m pip install --upgrade pip && \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 && \
pip install -r requirements.txt && \
python test_dataset_paths.py && \
python main.py train --epochs 200 --batch-size 24 --image-size 768 768 --experiment-name "runpod_auto_training"
```

## 💡 Pro Tips per RunPod

1. **Scegli GPU giusta**: RTX 4090 per max performance, RTX 3080 per budget
2. **Monitora costi**: Training completo costa ~$5-15 dipende da GPU e tempo
3. **Backup frequenti**: Scarica modelli intermedi ogni 50 epochs
4. **Multi-GPU**: Se disponibile, usa `--multi-gpu` per training parallelo
5. **Persistent Storage**: Salva in persistent volume se disponibile

Buona fortuna con il training su RunPod! 🚀
