"""
Script di setup per il progetto Star Removal
Installa tutte le dipendenze necessarie per il training della rete neurale
"""

import subprocess
import sys
import os

def install_dependencies():
    """Installa le dipendenze Python necessarie"""
    print("🚀 Installazione dipendenze per Star Removal Project...")
    
    # Aggiorna pip
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Installa PyTorch con supporto CUDA se disponibile
    try:
        import torch
        print(f"✅ PyTorch già installato - versione {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA disponibile - versione {torch.version.cuda}")
        else:
            print("⚠️  CUDA non disponibile, utilizzo CPU")
    except ImportError:
        print("📦 Installazione PyTorch...")
        # Installa versione con supporto CUDA per Windows
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "--index-url", 
            "https://download.pytorch.org/whl/cu118"
        ])
    
    # Installa altre dipendenze
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("✅ Tutte le dipendenze installate con successo!")

def check_system():
    """Controlla le specifiche del sistema"""
    print("\n🔍 Controllo sistema...")
    
    try:
        import torch
        print(f"✅ PyTorch versione: {torch.__version__}")
        print(f"✅ CUDA disponibile: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("❌ PyTorch non installato")
    
    try:
        import cv2
        print(f"✅ OpenCV versione: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV non installato")

if __name__ == "__main__":
    install_dependencies()
    check_system()