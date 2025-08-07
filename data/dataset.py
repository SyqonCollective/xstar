"""
Dataset personalizzato per training Star Removal
Include data augmentation avanzata per prevenire overfitting
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from typing import Tuple, List, Optional
import math

class StarRemovalDataset(Dataset):
    """
    Dataset per training rimozione stelle
    
    Features:
    - Data augmentation avanzata
    - Supporto per tile processing
    - Bilanciamento tra immagini 4K e piccole
    - Normalizzazione personalizzata per dati astronomici
    """
    
    def __init__(self, 
                 input_dir: str, 
                 starless_dir: str,
                 image_size: Tuple[int, int] = (512, 512),
                 augment: bool = True,
                 tile_overlap: float = 0.1,
                 use_tiles: bool = False,
                 augmentation_factor: int = 8):
        
        self.input_dir = Path(input_dir)
        self.starless_dir = Path(starless_dir)
        self.image_size = image_size
        self.augment = augment
        self.tile_overlap = tile_overlap
        self.use_tiles = use_tiles
        self.augmentation_factor = augmentation_factor
        
        # Trova tutte le coppie valide
        self.image_pairs = self._find_valid_pairs()
        
        # Se use_tiles, espandi dataset con tiles
        if use_tiles:
            self.image_pairs = self._create_tile_pairs()
        
        # Moltiplica dataset per augmentation factor
        self.image_pairs = self.image_pairs * augmentation_factor
        
        # Setup augmentation pipeline
        self.transform = self._get_transform()
        
        print(f"📊 Dataset inizializzato:")
        print(f"  - Coppie valide: {len(self.image_pairs) // augmentation_factor}")
        print(f"  - Con augmentation ({augmentation_factor}x): {len(self.image_pairs)}")
        print(f"  - Dimensione target: {image_size}")
        print(f"  - Usa tiles: {use_tiles}")
    
    def _find_valid_pairs(self) -> List[Tuple[str, str]]:
        """Trova coppie valide di immagini input-starless"""
        pairs = []
        
        for input_file in self.input_dir.glob("*.jpg"):
            starless_file = self.starless_dir / input_file.name
            
            if starless_file.exists():
                pairs.append((str(input_file), str(starless_file)))
        
        return pairs
    
    def _create_tile_pairs(self) -> List[Tuple[str, str, dict]]:
        """Crea tiles dalle immagini per aumentare il dataset"""
        tile_pairs = []
        
        for input_path, starless_path in self.image_pairs:
            # Leggi immagine per determinare dimensioni
            img = cv2.imread(input_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # Calcola numero di tiles
            tile_h, tile_w = self.image_size
            overlap_h = int(tile_h * self.tile_overlap)
            overlap_w = int(tile_w * self.tile_overlap)
            
            step_h = tile_h - overlap_h
            step_w = tile_w - overlap_w
            
            for y in range(0, h - tile_h + 1, step_h):
                for x in range(0, w - tile_w + 1, step_w):
                    tile_info = {
                        'x': x, 'y': y,
                        'w': tile_w, 'h': tile_h
                    }
                    tile_pairs.append((input_path, starless_path, tile_info))
        
        return tile_pairs if tile_pairs else [(p[0], p[1], None) for p in self.image_pairs]
    
    def _get_transform(self):
        """Crea pipeline di augmentation"""
        if not self.augment:
            return A.Compose([
                A.Resize(self.image_size[1], self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        # Augmentation astronomica specifica
        return A.Compose([
            # Augmentations geometriche
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=360, p=0.8),  # Rotazione completa per astronomia
                A.Transpose(p=0.3),
            ], p=0.8),
            
            # Augmentations fotometriche (delicate per astronomia)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.4
                ),
                A.RandomGamma(gamma_limit=(1.0, 1.2), p=0.4),  # Fixed: >= 1.0
            ], p=0.6),
            
            # Noise realistico per astronomia
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.4),
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.3), p=0.3),
            ], p=0.3),
            
            # Deformazioni elastiche leggere
            A.OneOf([
                A.ElasticTransform(alpha=0.5, sigma=5, alpha_affine=3, p=0.3),
                A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.2),
            ], p=0.2),
            
            # Resize e normalizzazione
            A.Resize(self.image_size[1], self.image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'starless': 'image'})
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        if self.use_tiles and len(self.image_pairs[idx]) == 3:
            input_path, starless_path, tile_info = self.image_pairs[idx]
        else:
            try:
                pair = self.image_pairs[idx]
                if len(pair) >= 2:
                    input_path, starless_path = pair[0], pair[1]
                else:
                    input_path = starless_path = pair[0] if pair else self.image_pairs[0][0]
            except IndexError:
                # Fallback: usa il primo elemento del dataset
                idx = idx % len(self.image_pairs)
                pair = self.image_pairs[idx]
                input_path, starless_path = pair[0], pair[1]
            tile_info = None
        
        # Carica immagini
        input_img = cv2.imread(input_path)
        starless_img = cv2.imread(starless_path)
        
        if input_img is None or starless_img is None:
            # Fallback a prima immagine valida, evita ricorsione infinita
            print(f"⚠️ Immagine corrotta: {input_path} o {starless_path}, salto...")
            # Prova la prossima immagine valida (max 10 tentativi)
            for fallback_idx in range(min(10, len(self.image_pairs))):
                try_idx = (idx + fallback_idx + 1) % len(self.image_pairs)
                if try_idx != idx:  # Evita loop infinito
                    fallback_pair = self.image_pairs[try_idx]
                    fallback_input = cv2.imread(fallback_pair[0])
                    fallback_starless = cv2.imread(fallback_pair[1])
                    if fallback_input is not None and fallback_starless is not None:
                        input_img, starless_img = fallback_input, fallback_starless
                        input_path, starless_path = fallback_pair[0], fallback_pair[1]
                        break
            else:
                # Se tutte le immagini sono corrotte, restituisci un'immagine dummy
                print(f"❌ Impossibile trovare immagini valide, creo dummy image")
                input_img = np.zeros((256, 256, 3), dtype=np.uint8)
                starless_img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Converti da BGR a RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        starless_img = cv2.cvtColor(starless_img, cv2.COLOR_BGR2RGB)
        
        # Estrai tile se specificato
        if tile_info is not None:
            x, y, w, h = tile_info['x'], tile_info['y'], tile_info['w'], tile_info['h']
            input_img = input_img[y:y+h, x:x+w]
            starless_img = starless_img[y:y+h, x:x+w]
        
        # Applica augmentation
        if self.augment:
            transformed = self.transform(image=input_img, starless=starless_img)
            input_img = transformed['image']
            starless_img = transformed['starless']
        else:
            transformed = self.transform(image=input_img)
            input_img = transformed['image']
            # Applica stessa trasformazione a starless
            starless_transformed = self.transform(image=starless_img)
            starless_img = starless_transformed['image']
        
        return {
            'input': input_img.float(),
            'starless': starless_img.float(),
            'filename': Path(input_path).stem
        }

class BalancedSampler:
    """
    Sampler per bilanciare immagini 4K e piccole durante il training
    """
    
    def __init__(self, dataset: StarRemovalDataset, ratio_4k: float = 0.7):
        self.dataset = dataset
        self.ratio_4k = ratio_4k
        
        # Categorizza immagini
        self.indices_4k = []
        self.indices_small = []
        
        for i, pair in enumerate(dataset.image_pairs):
            filename = Path(pair[0]).stem
            if filename.isdigit():
                self.indices_4k.append(i)
            else:
                self.indices_small.append(i)
        
        print(f"📊 Sampler bilanciato:")
        print(f"  - Indici 4K: {len(self.indices_4k)}")
        print(f"  - Indici piccole: {len(self.indices_small)}")
        print(f"  - Ratio 4K target: {ratio_4k}")
    
    def __iter__(self):
        # Calcola quanti esempi prendere da ogni categoria
        total_samples = len(self.dataset)
        n_4k = min(int(total_samples * self.ratio_4k), len(self.indices_4k))
        n_small = total_samples - n_4k
        
        # Sample con replacement se necessario
        sampled_4k = random.choices(self.indices_4k, k=n_4k) if self.indices_4k else []
        sampled_small = random.choices(self.indices_small, k=n_small) if self.indices_small else []
        
        # Mescola e restituisci
        all_indices = sampled_4k + sampled_small
        random.shuffle(all_indices)
        
        return iter(all_indices)
    
    def __len__(self):
        return len(self.dataset)

def create_dataloaders(input_dir: str, 
                      starless_dir: str,
                      batch_size: int = 8,
                      validation_split: float = 0.2,
                      image_size: Tuple[int, int] = (512, 512),
                      num_workers: int = 4,
                      use_tiles: bool = False,
                      augmentation_factor: int = 8) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoader per training e validation
    """
    
    # Dataset completo
    full_dataset = StarRemovalDataset(
        input_dir=input_dir,
        starless_dir=starless_dir,
        image_size=image_size,
        augment=True,
        use_tiles=use_tiles,
        augmentation_factor=augmentation_factor
    )
    
    # Split train/validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Dataset validation senza augmentation
    val_dataset_clean = StarRemovalDataset(
        input_dir=input_dir,
        starless_dir=starless_dir,
        image_size=image_size,
        augment=False,  # No augmentation per validation
        use_tiles=False,
        augmentation_factor=1
    )
    
    # Prendi subset per validation
    val_indices = val_dataset.indices if hasattr(val_dataset, 'indices') else list(range(val_size))
    val_subset = torch.utils.data.Subset(val_dataset_clean, val_indices[:val_size//augmentation_factor])
    
    # Crea DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"📊 DataLoaders creati:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_subset)}")
    print(f"  - Batch size: {batch_size}")
    
    return train_loader, val_loader

def test_dataset():
    """Test del dataset"""
    # Percorsi del dataset nel progetto
    input_dir = "../Starlossno-1/starmodel dataset/input"
    starless_dir = "../Starlossno-1/starmodel dataset/starless"
    
    # Test dataset base
    dataset = StarRemovalDataset(
        input_dir=input_dir,
        starless_dir=starless_dir,
        image_size=(256, 256),
        augment=True,
        augmentation_factor=2
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample
    sample = dataset[0]
    print(f"Input shape: {sample['input'].shape}")
    print(f"Starless shape: {sample['starless'].shape}")
    print(f"Filename: {sample['filename']}")
    
    # Test DataLoaders
    train_loader, val_loader = create_dataloaders(
        input_dir=input_dir,
        starless_dir=starless_dir,
        batch_size=2,
        image_size=(256, 256),
        augmentation_factor=2
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"Batch input shape: {batch['input'].shape}")
    print(f"Batch starless shape: {batch['starless'].shape}")

if __name__ == "__main__":
    test_dataset()