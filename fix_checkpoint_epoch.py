"""
Script per aggiornare il numero epoca nel checkpoint
"""

import torch

def fix_checkpoint_epoch(checkpoint_path):
    """Aggiorna il numero epoca nel checkpoint"""
    print(f"🔧 Caricando checkpoint: {checkpoint_path}")
    
    # Carica checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"📊 Epoca attuale nel checkpoint: {checkpoint.get('epoch', 'N/A')}")
    print(f"📊 Batch attuale nel checkpoint: {checkpoint.get('batch', 'N/A')}")
    
    # Aggiorna epoca a 2 (dato che è la fine della seconda epoca)
    checkpoint['epoch'] = 2
    
    # Rimuovi info batch per renderlo un checkpoint epoca pulito
    if 'batch' in checkpoint:
        del checkpoint['batch']
    
    print(f"✅ Aggiornato a epoca: {checkpoint['epoch']}")
    
    # Salva checkpoint aggiornato
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint salvato: {checkpoint_path}")

if __name__ == "__main__":
    fix_checkpoint_epoch("outputs/starnet_epoch2/checkpoint_epoch_2.pth")
