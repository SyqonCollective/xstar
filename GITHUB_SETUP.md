# 🚀 Quick Setup per GitHub

## 1. Preparazione Locale
```bash
# Esegui lo script di preparazione
./prepare_for_github.sh

# Verifica che tutto sia OK
python verify_github_ready.py
```

## 2. Setup Git e GitHub
```bash
# Inizializza Git (se non fatto)
git init

# Configura Git LFS per le immagini
git lfs install
git lfs track "*.jpg"
git lfs track "*.png"

# Aggiungi tutti i file
git add .

# Primo commit
git commit -m "Initial commit: Star Removal Project with dataset"

# Crea branch main
git branch -M main

# Collega al repository GitHub (sostituisci con il tuo URL)
git remote add origin https://github.com/TUO_USERNAME/star-removal-project.git

# Push su GitHub
git push -u origin main
```

## 3. Training su GitHub Actions
1. Vai su GitHub.com → Il tuo repository
2. Click su tab **"Actions"**
3. Seleziona **"Star Removal Training"**
4. Click **"Run workflow"**
5. Configura parametri:
   - Epochs: 50 (raccomandato per primi test)
   - Batch size: 2-4 (per CPU training)
   - Experiment name: "github_test_run"
6. Click **"Run workflow"**

## 4. Monitoraggio
- **Logs in tempo reale**: Actions tab → Run attivo
- **Download risultati**: Artifacts section al termine
- **TensorBoard**: Scarica tensorboard-logs artifact

## 5. Training Locale vs GitHub
### Locale (GPU raccomandato)
```bash
python main.py train --epochs 100 --batch-size 16 --image-size 512 512
```

### GitHub Actions (CPU ottimizzato)
- Automaticamente configurato per CPU
- Batch size ridotto (2-4)
- Image size ottimizzata (256x256)
- Early stopping attivo

## 6. Dopo il Training
I risultati saranno disponibili come GitHub Artifacts:
- `training-results`: Modello finale e metriche
- `tensorboard-logs`: Log per visualizzazione

## 🎯 Pro Tips
- **Dataset piccoli**: GitHub Actions funziona bene
- **Dataset grandi**: Considera Google Colab o Kaggle
- **GPU training**: Locale o cloud platforms
- **Monitoring**: GitHub Actions ha logs dettagliati

## ⚡ One-liner per setup completo
```bash
./prepare_for_github.sh && git add . && git commit -m "Ready for GitHub" && echo "Ora collega il repository remoto e fai git push!"
```
