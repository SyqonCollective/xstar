#!/usr/bin/env python3
"""
Test rapido per verificare che i percorsi del dataset siano corretti
"""
import os
import sys
from pathlib import Path

def test_dataset_paths():
    """Test dei percorsi del dataset"""
    print("🧪 Test percorsi dataset...")
    
    # Percorsi da testare
    input_dir = "Starlossno-1/starmodel dataset/input"
    starless_dir = "Starlossno-1/starmodel dataset/starless"
    
    # Verifica esistenza directory
    assert os.path.exists(input_dir), f"❌ Input directory non trovata: {input_dir}"
    assert os.path.exists(starless_dir), f"❌ Starless directory non trovata: {starless_dir}"
    
    print(f"✅ Input directory: {input_dir}")
    print(f"✅ Starless directory: {starless_dir}")
    
    # Conta immagini
    input_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    starless_images = [f for f in os.listdir(starless_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"📊 Input images: {len(input_images)}")
    print(f"📊 Starless images: {len(starless_images)}")
    
    # Verifica che il numero di immagini sia uguale
    assert len(input_images) == len(starless_images), f"❌ Numero immagini diverso: {len(input_images)} vs {len(starless_images)}"
    
    # Verifica corrispondenza nomi (optional)
    input_names = set(Path(f).stem for f in input_images)
    starless_names = set(Path(f).stem for f in starless_images)
    
    missing_starless = input_names - starless_names
    missing_input = starless_names - input_names
    
    if missing_starless:
        print(f"⚠️  Immagini starless mancanti: {missing_starless}")
    
    if missing_input:
        print(f"⚠️  Immagini input mancanti: {missing_input}")
    
    matching_pairs = len(input_names & starless_names)
    print(f"✅ Coppie corrispondenti: {matching_pairs}/{len(input_images)}")
    
    return len(input_images), len(starless_images), matching_pairs

def test_imports():
    """Test import dei moduli principali"""
    print("\n🐍 Test import moduli...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("⚠️  PyTorch non installato (necessario per training)")
        return False
    
    try:
        from data.dataset import StarRemovalDataset
        print("✅ Dataset module")
    except ImportError as e:
        print(f"❌ Errore import dataset: {e}")
        return False
    
    try:
        from models.unet_starnet import UNet
        print("✅ Model module")
    except ImportError as e:
        print(f"❌ Errore import model: {e}")
        return False
    
    return True

def test_dataset_loading():
    """Test caricamento dataset"""
    print("\n📦 Test caricamento dataset...")
    
    try:
        # Import necessari
        import sys
        sys.path.append('data')
        from dataset import StarRemovalDataset
        
        # Crea dataset
        dataset = StarRemovalDataset(
            input_dir="Starlossno-1/starmodel dataset/input",
            starless_dir="Starlossno-1/starmodel dataset/starless",
            image_size=(256, 256),  # Ridotto per test
            augment=False
        )
        
        print(f"✅ Dataset creato: {len(dataset)} samples")
        
        # Test caricamento primo sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample caricato: input shape {sample['input'].shape}, starless shape {sample['starless'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore caricamento dataset: {e}")
        return False

def main():
    print("🚀 Test configurazione XStar Dataset")
    print("=" * 50)
    
    try:
        # Test 1: Percorsi dataset
        input_count, starless_count, matching_pairs = test_dataset_paths()
        
        # Test 2: Import moduli  
        imports_ok = test_imports()
        
        # Test 3: Caricamento dataset (se PyTorch disponibile)
        if imports_ok:
            dataset_ok = test_dataset_loading()
        else:
            dataset_ok = False
            print("\n⚠️  Saltando test dataset (PyTorch non disponibile)")
        
        # Riassunto
        print("\n" + "=" * 50)
        print("📋 RIASSUNTO TEST:")
        print(f"✅ Dataset trovato: {input_count} coppie di immagini")
        print(f"{'✅' if matching_pairs == input_count else '⚠️ '} Coppie corrispondenti: {matching_pairs}/{input_count}")
        print(f"{'✅' if imports_ok else '❌'} Import moduli: {'OK' if imports_ok else 'ERRORI'}")
        print(f"{'✅' if dataset_ok else '⚠️ '} Caricamento dataset: {'OK' if dataset_ok else 'SKIP/ERRORE'}")
        
        if input_count > 0 and matching_pairs == input_count:
            print("\n🎉 CONFIGURAZIONE PRONTA PER TRAINING!")
            print("💡 Comando suggerito:")
            print("   python main.py train --epochs 50 --batch-size 4")
        else:
            print("\n⚠️  CONFIGURAZIONE INCOMPLETA")
            print("💡 Verifica che tutte le coppie di immagini siano presenti")
            
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FALLITO: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERRORE IMPREVISTO: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
