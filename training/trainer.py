"""
Training pipeline per Star Removal Network
Include validazione, early stopping e prevenzione overfitting
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.unet_starnet import StarNetUNet, StarNetLoss
from data.dataset import create_dataloaders

class EarlyStopping:
    """Early stopping per prevenire overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Controlla se fare early stopping
        Returns: True se dovremmo fermare il training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False

class MetricsTracker:
    """Tracker per metriche di training"""
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
    def update(self, epoch_metrics: Dict[str, float]):
        """Aggiorna metriche per epoch"""
        for key, value in epoch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_loss', minimize: bool = True) -> int:
        """Trova miglior epoch per una metrica"""
        if metric not in self.metrics or not self.metrics[metric]:
            return 0
        
        values = self.metrics[metric]
        if minimize:
            return np.argmin(values)
        else:
            return np.argmax(values)
    
    def save(self, filepath: str):
        """Salva metriche su file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(self, save_path: str):
        """Crea plot delle metriche"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[0, 1].plot(self.metrics['learning_rate'], alpha=0.8)
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epoch time plot
        axes[1, 0].plot(self.metrics['epoch_time'], alpha=0.8)
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Convergence plot (loss ratio)
        if len(self.metrics['val_loss']) > 1:
            val_loss = np.array(self.metrics['val_loss'])
            train_loss = np.array(self.metrics['train_loss'])
            ratio = val_loss / (train_loss + 1e-8)
            axes[1, 1].plot(ratio, alpha=0.8)
            axes[1, 1].set_title('Overfitting Monitor (Val/Train Loss)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Ratio')
            axes[1, 1].axhline(y=1.1, color='r', linestyle='--', alpha=0.5, label='Overfitting Threshold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

class StarRemovalTrainer:
    """
    Trainer principale per Star Removal Network
    """
    
    def __init__(self,
                 model: StarNetUNet,
                 train_loader,
                 val_loader,
                 device: str = 'cuda',
                 output_dir: str = 'outputs',
                 experiment_name: str = 'starnet_experiment'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loss, optimizer, scheduler
        self.criterion = StarNetLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Scheduler con warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart ogni 10 epoch
            T_mult=2,  # Raddoppia periodo ad ogni restart
            eta_min=1e-6
        )
        
        # Tracking
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(patience=15, min_delta=0.001)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.start_batch = 0
        
        # Prova a caricare checkpoint esistente
        self.load_latest_checkpoint()
        
        print(f"🚀 Trainer inizializzato:")
        print(f"  - Device: {device}")
        print(f"  - Output dir: {self.output_dir}")
        print(f"  - Parametri modello: {sum(p.numel() for p in model.parameters()):,}")
    
    def load_latest_checkpoint(self):
        """Carica l'ultimo checkpoint disponibile"""
        # Cerca checkpoint batch
        batch_checkpoints = list(self.output_dir.glob('checkpoint_batch_*.pth'))
        if batch_checkpoints:
            # Trova l'ultimo checkpoint batch
            latest_batch = max(batch_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            print(f"🔄 Trovato checkpoint batch: {latest_batch}")
            
            checkpoint = torch.load(latest_batch, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.start_batch = checkpoint['batch']
            
            print(f"📥 Checkpoint caricato: Epoch {self.current_epoch}, Batch {self.start_batch}")
            return True
            
        # Cerca checkpoint epoch
        epoch_checkpoints = list(self.output_dir.glob('checkpoint_epoch_*.pth'))
        if epoch_checkpoints:
            latest_epoch = max(epoch_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            print(f"🔄 Trovato checkpoint epoch: {latest_epoch}")
            
            checkpoint = torch.load(latest_epoch, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch'] + 1  # Prossima epoch
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            print(f"📥 Checkpoint caricato: riprende da Epoca {self.current_epoch} (completata epoca {checkpoint['epoch']})")
            return True
            
        print("🆕 Nessun checkpoint trovato, inizio da zero")
        return False
    
    def train_epoch(self) -> Dict[str, float]:
        """Training per una epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {'l1': 0, 'l2': 0, 'perceptual': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Salta batch già processati se riprende da checkpoint
            if batch_idx < self.start_batch:
                continue
            inputs = batch['input'].to(self.device)
            targets = batch['starless'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calcola loss
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumula metriche
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            
            # Salvataggio intermedio ogni 500 batch
            if (batch_idx + 1) % 500 == 0:
                checkpoint = {
                    'epoch': self.current_epoch,
                    'batch': batch_idx + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss.item(),
                    'metrics': loss_dict
                }
                checkpoint_path = self.output_dir / f'checkpoint_batch_{batch_idx+1}.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"💾 Checkpoint salvato: {checkpoint_path}")
            
            # Aggiorna progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'l1': f"{loss_dict['l1']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log batch metrics to tensorboard
            if batch_idx % 10 == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Batch_Loss', loss.item(), step)
                self.writer.add_scalar('Train/Learning_Rate', 
                                     self.optimizer.param_groups[0]['lr'], step)
        
        # Reset start_batch dopo la prima epoch
        self.start_batch = 0
        
        # Calcola metriche medie
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {k: v / len(self.train_loader) for k, v in loss_components.items()}
        
        return {'loss': avg_loss, **avg_components}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validazione per una epoch"""
        self.model.eval()
        total_loss = 0
        loss_components = {'l1': 0, 'l2': 0, 'perceptual': 0}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            
            for batch in pbar:
                inputs = batch['input'].to(self.device)
                targets = batch['starless'].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets)
                
                # Accumula metriche
                total_loss += loss.item()
                for key in loss_components:
                    loss_components[key] += loss_dict[key]
                
                pbar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'val_l1': f"{loss_dict['l1']:.4f}"
                })
        
        # Calcola metriche medie
        avg_loss = total_loss / len(self.val_loader)
        avg_components = {k: v / len(self.val_loader) for k, v in loss_components.items()}
        
        return {'loss': avg_loss, **avg_components}
    
    def save_sample_predictions(self, num_samples: int = 4):
        """Salva predizioni di esempio"""
        self.model.eval()
        
        with torch.no_grad():
            batch = next(iter(self.val_loader))
            inputs = batch['input'][:num_samples].to(self.device)
            targets = batch['starless'][:num_samples].to(self.device)
            filenames = batch['filename'][:num_samples]
            
            outputs = self.model(inputs)
            
            # Denormalizza per visualizzazione
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            
            inputs_vis = inputs * std + mean
            targets_vis = targets * std + mean
            outputs_vis = outputs * std + mean
            
            # Clamp values
            inputs_vis = torch.clamp(inputs_vis, 0, 1)
            targets_vis = torch.clamp(targets_vis, 0, 1)
            outputs_vis = torch.clamp(outputs_vis, 0, 1)
            
            # Crea griglia di visualizzazione
            fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
            
            for i in range(num_samples):
                # Input
                img_input = inputs_vis[i].cpu().permute(1, 2, 0).numpy()
                axes[0, i].imshow(img_input)
                axes[0, i].set_title(f'Input: {filenames[i]}')
                axes[0, i].axis('off')
                
                # Target
                img_target = targets_vis[i].cpu().permute(1, 2, 0).numpy()
                axes[1, i].imshow(img_target)
                axes[1, i].set_title('Target (Starless)')
                axes[1, i].axis('off')
                
                # Prediction
                img_pred = outputs_vis[i].cpu().permute(1, 2, 0).numpy()
                axes[2, i].imshow(img_pred)
                axes[2, i].set_title('Prediction')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            sample_path = self.output_dir / f'predictions_epoch_{self.current_epoch+1}.png'
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to tensorboard
            self.writer.add_figure('Predictions/Samples', fig, self.current_epoch)
    
    def save_checkpoint(self, is_best: bool = False):
        """Salva checkpoint del modello"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': self.metrics_tracker.metrics
        }
        
        # Salva checkpoint corrente
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Salva miglior modello
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Nuovo miglior modello salvato: {best_path}")
    
    def train(self, num_epochs: int = 100, save_every: int = 1):
        """Training loop principale"""
        print(f"🚀 Inizio training per {num_epochs} epochs...")
        
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Calcola tempo epoch
            epoch_time = time.time() - epoch_start
            
            # Tracking metriche
            epoch_metrics = {
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            self.metrics_tracker.update(epoch_metrics)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check se è il miglior modello
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Salva checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(is_best)
            
            # Salva predizioni di esempio
            if (epoch + 1) % (save_every * 2) == 0:
                self.save_sample_predictions()
            
            # Early stopping check
            if self.early_stopping(val_metrics['loss'], self.model):
                print(f"🛑 Early stopping attivato all'epoca {epoch+1}")
                break
            
            # Log progresso
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Time: {epoch_time:.1f}s")
            print("-" * 50)
        
        # Salva metriche finali
        self.metrics_tracker.save(self.output_dir / 'training_metrics.json')
        self.metrics_tracker.plot_metrics(self.output_dir / 'training_plots.png')
        
        # Carica miglior modello
        best_checkpoint = torch.load(self.output_dir / 'best_model.pth')
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        print(f"✅ Training completato!")
        print(f"📊 Miglior validation loss: {self.best_val_loss:.6f}")
        print(f"💾 Modelli salvati in: {self.output_dir}")
        
        self.writer.close()

def create_trainer(input_dir: str,
                  starless_dir: str,
                  output_dir: str = 'outputs',
                  experiment_name: str = 'starnet_experiment',
                  batch_size: int = 8,
                  image_size: Tuple[int, int] = (512, 512),
                  num_workers: int = 4,
                  device: str = None) -> StarRemovalTrainer:
    """Factory function per creare trainer"""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔧 Configurando trainer...")
    print(f"  Device: {device}")
    
    # Crea modello
    model = StarNetUNet(n_channels=3, n_classes=3, dropout_rate=0.1)
    
    # Crea dataloaders
    train_loader, val_loader = create_dataloaders(
        input_dir=input_dir,
        starless_dir=starless_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        use_tiles=True,  # Usa tiles per aumentare dataset
        augmentation_factor=8  # Ridotto a 8x per velocità simile al locale
    )
    
    # Crea trainer
    trainer = StarRemovalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        experiment_name=experiment_name
    )
    
    return trainer

if __name__ == "__main__":
    # Test trainer
    input_dir = "../Starlossno-1/starmodel dataset/input"
    starless_dir = "../Starlossno-1/starmodel dataset/starless"
    
    trainer = create_trainer(
        input_dir=input_dir,
        starless_dir=starless_dir,
        batch_size=4,
        image_size=(256, 256)
    )
    
    # Training di test breve
    trainer.train(num_epochs=2, save_every=1)