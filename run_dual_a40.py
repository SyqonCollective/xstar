#!/usr/bin/env python3
"""
Training ottimizzato per RunPod 2x A40 (96GB VRAM totale)
Utilizza DataParallel per sfruttare entrambe le GPU
"""

import torch
import torch.nn as nn
from training.trainer import create_trainer

def main():
    print("🚀 Dual A40 Training Setup (96GB VRAM)")
    print("=" * 60)
    
    # 🔥 VERIFICHE GPU CRITICHE
    if not torch.cuda.is_available():
        print("❌ CUDA non disponibile!")
        return
        
    gpu_count = torch.cuda.device_count()
    print(f"🎯 GPU Count: {gpu_count}")
    
    total_memory = 0
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        total_memory += gpu_memory
        print(f"🔥 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print(f"🚀 Total GPU Memory: {total_memory:.1f}GB")
    
    # 🔥 CONFIGURAZIONE OTTIMIZZATA per 2x A40
    is_dual_gpu = gpu_count >= 2
    
    # Batch size aggressivo per 96GB VRAM
    if is_dual_gpu and total_memory >= 90:
        batch_size = 32  # 16 per GPU con 96GB totale
        print("🚀 DUAL A40 DETECTED - Using aggressive batch size!")
    elif total_memory >= 40:
        batch_size = 20  # Single A40
    else:
        batch_size = 12  # Fallback
    
    config = {
        "input_dir": "Starlossno-1/starmodel dataset/input",
        "starless_dir": "Starlossno-1/starmodel dataset/starless", 
        "output_dir": "outputs",
        "experiment_name": f"dual_a40_training" if is_dual_gpu else "a40_training",
        "batch_size": batch_size,
        "image_size": (512, 512),  # Può andare fino a 768x768 con 96GB
        "num_workers": 16 if is_dual_gpu else 12,  # Più workers per dual GPU
        "device": "cuda"
    }
    
    print(f"📊 Training Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    
    # Crea trainer con ottimizzazioni
    trainer = create_trainer(**config)
    
    # 🚀 SETUP MULTI-GPU se disponibile
    if is_dual_gpu:
        print("🔥 Enabling DataParallel for Dual GPU!")
        trainer.model = nn.DataParallel(trainer.model)
        
        # Ottimizzazioni specifiche per multi-GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("✅ Multi-GPU setup complete!")
    
    # Avvia training con parametri ottimizzati
    print("🚀 Starting optimized training...")
    trainer.train(num_epochs=150, save_every=5)  # Più epoche con GPU potenti
    
    print("✅ Training completato!")

if __name__ == "__main__":
    main()
