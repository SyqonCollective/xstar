#!/usr/bin/env python3
"""
Training PROFESSIONALE per 70 epoche con MASSIMA AUGMENTATION
Ottimizzato per 2x A40 96GB VRAM - Anti-Overfitting
"""

import torch
import torch.nn as nn
from pathlib import Path
from data.dataset import create_dataloaders
from training.trainer import StarNetTrainer
from models.unet_starnet import UNet

def main():
    print("🚀 PROFESSIONAL 70-EPOCH TRAINING - 2x A40 OPTIMIZED")
    print("🔥 MAXIMUM AUGMENTATION + ANTI-OVERFITTING")
    print("=" * 70)
    
    # GPU Setup
    device_count = torch.cuda.device_count()
    total_memory = sum(torch.cuda.get_device_properties(i).total_memory / 1e9 
                      for i in range(device_count))
    
    print(f"🎯 GPU Count: {device_count}")
    print(f"🚀 Total VRAM: {total_memory:.1f}GB")
    
    # 🔥 CONFIGURAZIONE PROFESSIONALE 70 EPOCHE
    config = {
        # Dataset paths
        "input_dir": "Starlossno-1/starmodel dataset/input",
        "starless_dir": "Starlossno-1/starmodel dataset/starless",
        "output_dir": "outputs/professional_70epochs",
        
        # Training parameters
        "num_epochs": 70,
        "batch_size": 28 if total_memory >= 90 else 20,  # Ottimale per dual A40
        "learning_rate": 1e-4,  # Più basso per training lungo
        "validation_split": 0.25,  # Più validation per anti-overfitting
        
        # Image processing
        "image_size": (512, 512),
        "use_tiles": False,  # Evita overfitting su tiles
        
        # 🚨 MASSIMA AUGMENTATION (key anti-overfitting)
        "augmentation_factor": 12,  # 12x augmentation!
        
        # DataLoader optimization
        "num_workers": 20 if device_count >= 2 else 12,
        "pin_memory": True,
        "persistent_workers": True,
        
        # Anti-overfitting features
        "dropout_rate": 0.3,  # Dropout alto
        "weight_decay": 1e-3,  # Regolarizzazione forte
        "early_stopping_patience": 12,  # Stop se non migliora per 12 epoche
        
        # Checkpointing
        "save_every": 5,
        "validate_every": 2,
    }
    
    print("📊 PROFESSIONAL CONFIGURATION:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    
    # Crea output directory
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # 🔥 DATALOADERS CON MASSIMA AUGMENTATION
    print("🔧 Creating MAXIMUM AUGMENTATION DataLoaders...")
    train_loader, val_loader = create_dataloaders(
        input_dir=config["input_dir"],
        starless_dir=config["starless_dir"],
        batch_size=config["batch_size"],
        validation_split=config["validation_split"],
        image_size=config["image_size"],
        num_workers=config["num_workers"],
        use_tiles=config["use_tiles"],
        augmentation_factor=config["augmentation_factor"]  # 12x augmentation!
    )
    
    print(f"✅ DataLoaders created:")
    print(f"  Training samples: {len(train_loader.dataset)} (with {config['augmentation_factor']}x augmentation)")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Batch size: {config['batch_size']}")
    
    # 🔥 MODELLO CON ANTI-OVERFITTING
    print("🔧 Creating model with ANTI-OVERFITTING features...")
    model = UNet(
        in_channels=3,
        out_channels=3,
        features=[64, 128, 256, 512],  # Architettura robusta
        dropout_rate=config["dropout_rate"]  # Dropout per anti-overfitting
    )
    
    # Multi-GPU setup
    if device_count >= 2:
        print("🚀 Enabling DataParallel for DUAL A40!")
        model = nn.DataParallel(model)
        
        # A40 specific optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    model = model.cuda()
    
    # 🔥 TRAINER CON ANTI-OVERFITTING
    trainer = StarNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],  # Strong regularization
        device="cuda"
    )
    
    # 🚨 TRAINING PROFESSIONALE 70 EPOCHE
    print("🚀 Starting PROFESSIONAL 70-EPOCH TRAINING...")
    print("🔥 Features:")
    print("  ✅ 12x Data Augmentation (anti-overfitting)")
    print("  ✅ High Dropout (0.3)")
    print("  ✅ Strong Weight Decay (1e-3)")  
    print("  ✅ Large Validation Split (25%)")
    print("  ✅ Early Stopping (12 patience)")
    print("  ✅ Multi-GPU DataParallel")
    print("  ✅ Dual A40 Optimizations")
    print("=" * 70)
    
    # Avvia training
    trainer.train(
        num_epochs=config["num_epochs"],
        save_every=config["save_every"],
        validate_every=config["validate_every"]
    )
    
    print("✅ PROFESSIONAL 70-EPOCH TRAINING COMPLETED!")
    print(f"📁 Results saved in: {config['output_dir']}")

if __name__ == "__main__":
    main()
