"""
Test del modello epoca 2 per vedere i miglioramenti
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from models.unet_starnet import StarNetUNet
from torchvision import transforms

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Carica il modello dal checkpoint"""
    print(f"🔄 Caricando modello da: {checkpoint_path}")
    
    # Crea modello
    model = StarNetUNet(n_channels=3, n_classes=3)
    
    # Carica checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modello caricato (Epoca {checkpoint.get('epoch', 'N/A')}, Batch {checkpoint.get('batch', 'N/A')})")
    return model

def process_image(model, image_path, device='cuda'):
    """Processa una singola immagine"""
    # Carica immagine
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossibile caricare: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img_rgb.shape[:2]
    
    # Preprocessing (come nel training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Resize a 256x256 per il modello
    img_resized = cv2.resize(img_rgb, (256, 256))
    input_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    # Inferenza
    with torch.no_grad():
        output = model(input_tensor)
    
    # Post-processing
    output = output.squeeze(0).cpu()
    
    # Denormalizza
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    output = output * std + mean
    output = torch.clamp(output, 0, 1)
    
    # Converti a numpy e resize alla dimensione originale
    output_np = output.permute(1, 2, 0).numpy()
    output_resized = cv2.resize(output_np, (original_size[1], original_size[0]))
    
    return img_rgb, output_resized

def create_comparison(original, processed, save_path):
    """Crea una visualizzazione comparativa"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Immagine originale
    axes[0].imshow(original)
    axes[0].set_title('Originale (con stelle)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Immagine processata
    axes[1].imshow(processed)
    axes[1].set_title('Epoca 2 (stelle rimosse)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"💾 Confronto salvato: {save_path}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Usando device: {device}")
    
    # Path del checkpoint più recente della seconda epoca
    checkpoint_path = "outputs/starnet_epoch2/checkpoint_batch_9500.pth"
    
    # Carica modello
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # Immagini di test disponibili
    test_images = [
        "test_images/galaxy.jpg",
        "test_input/whirlpool_galaxy.jpg"
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n🎨 Processando: {img_path}")
            
            try:
                # Processa immagine
                original, processed = process_image(model, img_path, device)
                
                # Salva risultato
                output_name = f"epoch2_result_{Path(img_path).stem}.jpg"
                output_path = f"galaxy_results_epoch2/{output_name}"
                Path("galaxy_results_epoch2").mkdir(exist_ok=True)
                
                # Salva immagine processata
                processed_bgr = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, processed_bgr)
                
                # Crea confronto
                comparison_path = f"galaxy_results_epoch2/comparison_{Path(img_path).stem}.png"
                create_comparison(original, processed, comparison_path)
                
                print(f"✅ Risultato salvato: {output_path}")
                
            except Exception as e:
                print(f"❌ Errore processando {img_path}: {e}")
        else:
            print(f"⚠️ File non trovato: {img_path}")

if __name__ == "__main__":
    main()
