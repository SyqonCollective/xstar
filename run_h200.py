#!/usr/bin/env python3
"""
Quick start ottimizzato per RunPod H200
Risolve automaticamente il problema 0% GPU usage
"""

import torch
import os
from training.trainer import create_trainer

def main():
    print("🚀 H200 Training Setup")
    print("=" * 50)
    
    # 🔥 H200 SPECIFIC CHECKS
    if not torch.cuda.is_available():
        print("❌ CUDA non disponibile!")
        return
        
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🎯 GPU Detected: {gpu_name}")
    
    # H200 detection
    is_h200 = "H200" in gpu_name.upper()
    if is_h200:
        print("🚀 H200 DETECTED - Applying optimizations!")
    
    # 🔥 CONFIGURAZIONE OTTIMIZZATA PER H200
    config = {
        "input_dir": "Starlossno-1/starmodel dataset/input",
        "starless_dir": "Starlossno-1/starmodel dataset/starless", 
        "output_dir": "outputs",
        "experiment_name": "h200_optimized" if is_h200 else "gpu_training",
        "batch_size": 24 if is_h200 else 16,  # H200 può gestire batch più grandi
        "image_size": (512, 512),
        "num_workers": 16,  # Sarà ottimizzato automaticamente
        "device": "cuda"
    }
    
    print(f"📊 Training Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    # Crea trainer (con tutte le ottimizzazioni H200 già integrate)
    trainer = create_trainer(**config)
    
    # Avvia training
    trainer.train(num_epochs=100, save_every=5)
    
    print("✅ Training completato!")

if __name__ == "__main__":
    main()
