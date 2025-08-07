# 🚫 GitHub Actions Training DISABLED

## Questo repository NON esegue training su GitHub Actions

**Perché disabilitato:**
- GitHub Actions non ha GPU
- Training CPU è inutilmente lento 
- Costa risorse inutili
- Genera spam di email

## ✅ Come fare training

**Usa RunPod o hardware locale con GPU:**

1. **RunPod (Raccomandato):**
   ```bash
   # Segui la guida completa in RUNPOD_TRAINING_GUIDE.md
   git clone https://github.com/SyqonCollective/xstar.git
   cd xstar
   python main.py train --epochs 80 --batch-size 16 --image-size 768 768
   ```

2. **Locale con GPU:**
   ```bash
   git clone https://github.com/SyqonCollective/xstar.git  
   cd xstar
   pip install -r requirements.txt
   python main.py train --epochs 200 --batch-size 8 --image-size 640 640
   ```

## 📖 Documentazione

- `RUNPOD_TRAINING_GUIDE.md` - Guida completa RunPod
- `README.md` - Overview del progetto
- `main.py train --help` - Opzioni training

**Repository:** https://github.com/SyqonCollective/xstar
