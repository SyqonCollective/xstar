# 🌟 Star Removal Project

Un sistema completo di deep learning per la rimozione delle stelle da immagini astronomiche, ispirato a StarNet e StarXterminator.

## 🎯 Caratteristiche Principali

- **Architettura U-Net Avanzata**: Con attention mechanism e skip connections
- **Data Augmentation Astronomica**: Augmentazioni specifiche per dati astronomici
- **Tile Processing**: Elaborazione di immagini ad alta risoluzione
- **Prevenzione Overfitting**: Early stopping, dropout, e tecniche avanzate
- **Metriche Specializzate**: Valutazione specifica per rimozione stelle
- **Pipeline Completa**: Da analisi dati a inferenza finale

## 🚀 Quick Start

### Installazione Rapida
```bash
cd star_removal_project
python setup.py
```

### Avvio Immediato
```bash
python quick_start.py
```

Il quick start ti guiderà attraverso:
1. ✅ Installazione dipendenze
2. 🔍 Analisi dataset
3. 🚀 Training del modello
4. 📊 Valutazione performance
5. 🎨 Processamento immagini

## 📋 Requisiti

### Hardware Raccomandato
- **GPU**: NVIDIA con almeno 6GB VRAM (RTX 3060 o superiore)
- **RAM**: Almeno 16GB
- **Storage**: 10GB liberi per dataset e modelli

### Software
- Python 3.8+
- CUDA 11.8+ (per GPU)
- PyTorch 2.0+

## 📁 Struttura Progetto

```
star_removal_project/
├── data/                   # Gestione dataset e augmentation
├── models/                 # Architetture U-Net
├── training/              # Pipeline di training
├── inference/             # Tile processing e inferenza
├── evaluation/            # Metriche e valutazione
├── outputs/               # Risultati training
├── main.py               # Script principale
├── quick_start.py        # Avvio rapido
└── README.md             # Questo file
```

## 🔧 Utilizzo Avanzato

### 1. Analisi Dataset
```bash
python main.py analyze --input-dir /path/to/input --starless-dir /path/to/starless
```

### 2. Training Personalizzato
```bash
python main.py train \
    --epochs 200 \
    --batch-size 8 \
    --image-size 512 512 \
    --experiment-name my_experiment
```

### 3. Valutazione Modello
```bash
python main.py evaluate \
    --model-path outputs/my_experiment/best_model.pth \
    --input-dir /path/to/test/input \
    --starless-dir /path/to/test/starless
```

### 4. Processamento Immagini
```bash
python main.py process \
    --model-path outputs/my_experiment/best_model.pth \
    --input-dir /path/to/images \
    --output-dir results \
    --tile-size 1024 1024 \
    --overlap 0.25
```

### 5. Pipeline Completa
```bash
python main.py full --epochs 100 --batch-size 8
```

## 🏗️ Architettura del Modello

### U-Net con Attention
- **Encoder**: Downsampling con DoubleConv + MaxPool
- **Bottleneck**: Feature extraction profonda
- **Attention**: Focalizzazione automatica sulle stelle
- **Decoder**: Upsampling con skip connections
- **Output**: Connessioni residuali per preservare dettagli

### Loss Function Ibrida
- **L1 Loss**: Preservazione dettagli fini
- **L2 Loss**: Smoothness generale
- **Perceptual Loss**: Preservazione strutture semantiche

## 📊 Metriche di Valutazione

### Metriche Standard
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MSE/MAE**: Errori pixel-wise

### Metriche Astronomiche
- **Star Removal Rate**: Efficienza rimozione stelle
- **Background Preservation**: Preservazione sfondo
- **Gradient Preservation**: Conservazione dettagli fini
- **Color Preservation**: Mantenimento colori

## 🎛️ Configurazioni Ottimali

### Dataset Piccolo (< 100 immagini)
```bash
python main.py train \
    --epochs 150 \
    --batch-size 4 \
    --image-size 512 512
```

### Dataset Medio (100-500 immagini)
```bash
python main.py train \
    --epochs 100 \
    --batch-size 8 \
    --image-size 512 512
```

### Dataset Grande (> 500 immagini)
```bash
python main.py train \
    --epochs 80 \
    --batch-size 16 \
    --image-size 768 768
```

## 🔄 Data Augmentation

### Augmentazioni Geometriche
- Rotazioni complete (0-360°)
- Flip orizzontali/verticali
- Trasposizioni

### Augmentazioni Fotometriche (Conservative)
- Brightness/Contrast: ±10%
- Gamma correction: 0.9-1.1
- Hue/Saturation: ±5%

### Augmentazioni Rumore
- Gaussian noise realistico
- ISO noise simulation

## 🧪 Tile Processing

Per immagini grandi (> 2K):

```python
from inference.tile_processor import TileProcessor

processor = TileProcessor(
    model=model,
    tile_size=(1024, 1024),
    overlap=0.25,
    batch_size=4
)

result = processor.process_image("large_image.jpg", "output.jpg")
```

### Vantaggi Tile Processing
- ✅ Gestione immagini arbitrariamente grandi
- ✅ Uso efficiente memoria GPU
- ✅ Blending seamless senza artifacts
- ✅ Parallelizzazione automatica

## 📈 Monitoraggio Training

### TensorBoard
```bash
tensorboard --logdir outputs/experiment_name/tensorboard
```

### Metriche Monitorate
- Loss di training e validation
- Learning rate schedule
- Tempo per epoch
- Predizioni di esempio
- Overfitting detection

## 🎯 Risultati Attesi

### Performance Target
- **PSNR**: > 30 dB
- **SSIM**: > 0.95
- **Star Removal Rate**: > 0.90
- **Background Preservation**: > 0.95

### Tempi di Training
- **63 immagini**: ~2-4 ore (RTX 3080)
- **Con augmentation 8x**: ~504 campioni effettivi
- **Convergenza**: Tipicamente 50-100 epochs

## 🔧 Troubleshooting

### Memoria GPU Insufficiente
```bash
# Riduci batch size
python main.py train --batch-size 2

# Riduci dimensioni immagini
python main.py train --image-size 256 256
```

### Overfitting
- Il sistema include già early stopping
- Aumenta dropout rate nel modello
- Riduci learning rate

### Underfitting
- Aumenta model capacity
- Riduci dropout
- Aumenta learning rate iniziale

## 📚 Riferimenti Tecnici

### Ispirazione
- **StarNet**: Rimozione stelle con CNN
- **StarXterminator**: Processing astronomico avanzato
- **U-Net**: Architettura per segmentazione

### Papers Correlati
- "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- "Attention U-Net: Learning Where to Look for the Pancreas"
- "Deep Learning for Astronomical Image Processing"

## 🌐 Training su GitHub Actions

Questo progetto supporta il training automatico su GitHub Actions:

### Setup Rapido
1. **Fork** questo repository
2. Vai su **Actions** → **Star Removal Training**
3. Click **"Run workflow"**
4. Configura parametri:
   - `epochs`: Numero di epoche (default: 50)
   - `batch_size`: Dimensione batch (default: 4)
   - `experiment_name`: Nome esperimento

### Training Automatico
Il workflow GitHub Actions:
- ✅ Installa automaticamente le dipendenze
- ✅ Verifica il dataset
- ✅ Avvia training su CPU (ottimizzato per risorse limitate)
- ✅ Carica risultati come artifacts
- ✅ Genera logs TensorBoard

### Artifacts Disponibili
- `training-results`: Modelli e logs finali
- `tensorboard-logs`: Logs per visualizzazione metriche

## 🐳 Docker Support

### Build e Run
```bash
# Build immagine
docker build -t star-removal .

# Training con Docker
docker-compose up star-removal

# TensorBoard
docker-compose up tensorboard
# Vai su http://localhost:6006

# Jupyter Notebook
docker-compose up jupyter
# Vai su http://localhost:8888
```

## 🔧 Git LFS Setup

Per dataset grandi:
```bash
# Installa Git LFS
git lfs install

# Track immagini (già configurato in .gitattributes)
git lfs track "*.jpg"
git add .gitattributes
git commit -m "Add LFS tracking"
```

## 🚀 Deploy su Cloud

### Google Colab
```python
!git clone https://github.com/TUO_USERNAME/star-removal-project.git
%cd star-removal-project
!python setup.py
!python quick_start.py
```

### Kaggle Notebooks
Il progetto è ottimizzato per Kaggle con dataset incluso.

## 🤝 Contributi

Per miglioramenti o bug reports:
1. Crea issue descrittivo
2. Fork del repository
3. Sviluppa feature/fix
4. Pull request con descrizione
5. Il CI/CD testerà automaticamente le modifiche

## 📄 Licenza

MIT License - Progetto open source per ricerca e uso educativo in astronomia.

---

## 🎉 Conclusione

Questo progetto fornisce una soluzione completa e professionale per la rimozione delle stelle da immagini astronomiche. L'architettura U-Net con attention mechanism, combinata con tecniche avanzate di data augmentation e tile processing, garantisce risultati di alta qualità anche su dataset limitati.

**Buona fortuna con il tuo progetto di astrofotografia! 🌌**