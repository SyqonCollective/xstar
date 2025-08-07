"""
Tile processor per elaborazione di immagini ad alta risoluzione
Divide immagini grandi in tile, elabora separatamente e ricompone
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, List, Optional, Union
import math
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))

from models.unet_starnet import StarNetUNet

class TileProcessor:
    """
    Processore per elaborazione a tile di immagini grandi
    
    Features:
    - Suddivisione intelligente in tile con overlap
    - Blending seamless per evitare artifacts
    - Supporto per memoria limitata
    - Ottimizzazione per batch processing
    """
    
    def __init__(self,
                 model: torch.nn.Module,
                 tile_size: Tuple[int, int] = (512, 512),
                 overlap: float = 0.25,
                 device: str = 'cuda',
                 batch_size: int = 4,
                 normalize: bool = True):
        
        self.model = model.to(device)
        self.model.eval()
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Calcola dimensioni overlap
        self.overlap_w = int(tile_size[0] * overlap)
        self.overlap_h = int(tile_size[1] * overlap)
        
        # Step size (tile_size - overlap)
        self.step_w = tile_size[0] - self.overlap_w
        self.step_h = tile_size[1] - self.overlap_h
        
        print(f"🔧 TileProcessor inizializzato:")
        print(f"  - Tile size: {tile_size}")
        print(f"  - Overlap: {overlap} ({self.overlap_w}x{self.overlap_h} px)")
        print(f"  - Step size: {self.step_w}x{self.step_h}")
        print(f"  - Device: {device}")
        print(f"  - Batch size: {batch_size}")
    
    def _normalize_image(self, image: np.ndarray) -> torch.Tensor:
        """Normalizza immagine per il modello"""
        if self.normalize:
            # Converti a float e normalizza [0,1]
            image = image.astype(np.float32) / 255.0
            
            # Normalizzazione ImageNet
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            image = (image - mean) / std
        else:
            image = image.astype(np.float32) / 255.0
        
        # Converti a tensor PyTorch (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        return tensor
    
    def _denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Denormalizza tensor per output"""
        # (C, H, W) -> (H, W, C)
        image = tensor.permute(1, 2, 0).cpu().numpy()
        
        if self.normalize:
            # Denormalizzazione ImageNet
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            image = image * std + mean
        
        # Clamp e converti a uint8
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image
    
    def _create_blend_mask(self, tile_h: int, tile_w: int) -> torch.Tensor:
        """Crea maschera per blending seamless"""
        # Crea maschera con fadeout ai bordi
        mask = torch.ones(tile_h, tile_w)
        
        # Fadeout orizzontale
        if self.overlap_w > 0:
            fade_w = self.overlap_w // 2
            for i in range(fade_w):
                weight = i / fade_w
                mask[:, i] *= weight
                mask[:, -(i+1)] *= weight
        
        # Fadeout verticale
        if self.overlap_h > 0:
            fade_h = self.overlap_h // 2
            for i in range(fade_h):
                weight = i / fade_h
                mask[i, :] *= weight
                mask[-(i+1), :] *= weight
        
        return mask.to(self.device)
    
    def _calculate_tiles(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """Calcola posizioni di tutti i tile"""
        tiles = []
        
        # Calcola numero di tile necessari
        n_tiles_h = math.ceil((height - self.overlap_h) / self.step_h)
        n_tiles_w = math.ceil((width - self.overlap_w) / self.step_w)
        
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calcola posizione tile
                y_start = i * self.step_h
                x_start = j * self.step_w
                
                y_end = min(y_start + self.tile_size[1], height)
                x_end = min(x_start + self.tile_size[0], width)
                
                # Aggiusta posizione se il tile è più piccolo del previsto
                if y_end - y_start < self.tile_size[1]:
                    y_start = max(0, y_end - self.tile_size[1])
                if x_end - x_start < self.tile_size[0]:
                    x_start = max(0, x_end - self.tile_size[0])
                
                tiles.append((y_start, y_end, x_start, x_end))
        
        return tiles
    
    def _extract_tile(self, image: np.ndarray, tile_coords: Tuple[int, int, int, int]) -> torch.Tensor:
        """Estrae singolo tile dall'immagine"""
        y_start, y_end, x_start, x_end = tile_coords
        tile = image[y_start:y_end, x_start:x_end]
        
        # Pad se necessario per raggiungere tile_size
        current_h, current_w = tile.shape[:2]
        if current_h < self.tile_size[1] or current_w < self.tile_size[0]:
            pad_h = max(0, self.tile_size[1] - current_h)
            pad_w = max(0, self.tile_size[0] - current_w)
            
            # Padding riflesso per continuità
            tile = cv2.copyMakeBorder(
                tile, 0, pad_h, 0, pad_w, 
                cv2.BORDER_REFLECT_101
            )
        
        return self._normalize_image(tile)
    
    def _process_tile_batch(self, tile_batch: torch.Tensor) -> torch.Tensor:
        """Processa batch di tile con il modello"""
        with torch.no_grad():
            tile_batch = tile_batch.to(self.device).float()  # Forza tipo Float32
            output_batch = self.model(tile_batch)
            
            # Applica tanh se necessario (output in [-1, 1])
            if hasattr(self.model, 'outc'):
                output_batch = torch.tanh(output_batch)
            
            return output_batch
    
    def process_image(self, 
                     image: Union[str, np.ndarray],
                     output_path: Optional[str] = None,
                     show_progress: bool = True) -> np.ndarray:
        """
        Processa immagine completa usando tile processing
        
        Args:
            image: Path dell'immagine o array numpy
            output_path: Path per salvare risultato (opzionale)
            show_progress: Mostra progress bar
            
        Returns:
            Immagine processata come numpy array
        """
        
        # Carica immagine se è un path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Impossibile caricare immagine: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        
        height, width = img.shape[:2]
        
        print(f"🔍 Processando immagine {width}x{height}...")
        
        # Calcola tile positions
        tiles = self._calculate_tiles(height, width)
        n_tiles = len(tiles)
        
        print(f"📊 Divisa in {n_tiles} tile ({self.tile_size[0]}x{self.tile_size[1]})")
        
        # Prepara output
        output_img = np.zeros_like(img, dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Crea maschera per blending
        blend_mask = self._create_blend_mask(self.tile_size[1], self.tile_size[0])
        
        # Processa tile in batch
        tile_batches = [tiles[i:i+self.batch_size] for i in range(0, n_tiles, self.batch_size)]
        
        if show_progress:
            pbar = tqdm(tile_batches, desc="Processando tile")
        else:
            pbar = tile_batches
        
        for batch_tiles in pbar:
            # Estrai batch di tile
            tile_tensors = []
            for tile_coords in batch_tiles:
                tile_tensor = self._extract_tile(img, tile_coords)
                tile_tensors.append(tile_tensor)
            
            # Stack in batch
            tile_batch = torch.stack(tile_tensors)
            
            # Processa batch
            output_batch = self._process_tile_batch(tile_batch)
            
            # Ricomponi risultati
            for i, (tile_coords, output_tile) in enumerate(zip(batch_tiles, output_batch)):
                y_start, y_end, x_start, x_end = tile_coords
                
                # Denormalizza tile
                processed_tile = self._denormalize_image(output_tile)
                
                # Gestisci dimensioni (rimuovi padding se aggiunto)
                actual_h = y_end - y_start
                actual_w = x_end - x_start
                processed_tile = processed_tile[:actual_h, :actual_w]
                
                # Applica blending mask
                mask = blend_mask[:actual_h, :actual_w].cpu().numpy()
                
                # Accumula nel output con peso
                output_img[y_start:y_end, x_start:x_end] += processed_tile * mask[:, :, np.newaxis]
                weight_map[y_start:y_end, x_start:x_end] += mask
        
        # Normalizza per peso
        weight_map = np.maximum(weight_map, 1e-8)  # Evita divisione per zero
        output_img = output_img / weight_map[:, :, np.newaxis]
        
        # Converti a uint8
        result = np.clip(output_img, 0, 255).astype(np.uint8)
        
        # Salva se richiesto
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Converti RGB -> BGR per OpenCV
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), result_bgr)
            print(f"💾 Risultato salvato: {output_path}")
        
        return result
    
    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         pattern: str = "*.jpg",
                         show_progress: bool = True):
        """
        Processa tutte le immagini in una directory
        
        Args:
            input_dir: Directory input
            output_dir: Directory output
            pattern: Pattern per filtrare file
            show_progress: Mostra progress bar
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Trova tutti i file
        image_files = list(input_path.glob(pattern))
        
        if not image_files:
            print(f"❌ Nessuna immagine trovata in {input_dir} con pattern {pattern}")
            return
        
        print(f"🔍 Trovate {len(image_files)} immagini da processare")
        
        if show_progress:
            pbar = tqdm(image_files, desc="Processando immagini")
        else:
            pbar = image_files
        
        for img_file in pbar:
            try:
                # Genera path output
                output_file = output_path / f"starless_{img_file.name}"
                
                # Processa immagine
                self.process_image(
                    str(img_file),
                    str(output_file),
                    show_progress=False  # No progress per singola immagine
                )
                
                if show_progress:
                    pbar.set_postfix({'current': img_file.name})
                    
            except Exception as e:
                print(f"❌ Errore processando {img_file.name}: {e}")
                continue
        
        print(f"✅ Processamento completato. Risultati in: {output_path}")

def benchmark_tile_processor(model_path: str, test_image_path: str):
    """Benchmark del tile processor con diverse configurazioni"""
    
    print("🚀 Benchmark TileProcessor...")
    
    # Carica modello
    model = StarNetUNet(n_channels=3, n_classes=3)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test diverse configurazioni
    configs = [
        {"tile_size": (256, 256), "overlap": 0.1, "batch_size": 8},
        {"tile_size": (512, 512), "overlap": 0.2, "batch_size": 4},
        {"tile_size": (1024, 1024), "overlap": 0.25, "batch_size": 2},
    ]
    
    for config in configs:
        print(f"\n📊 Testando configurazione: {config}")
        
        processor = TileProcessor(
            model=model,
            tile_size=config["tile_size"],
            overlap=config["overlap"],
            batch_size=config["batch_size"]
        )
        
        import time
        start_time = time.time()
        
        result = processor.process_image(test_image_path, show_progress=False)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"  ⏱️  Tempo: {processing_time:.2f}s")
        print(f"  📐 Output shape: {result.shape}")

if __name__ == "__main__":
    # Test del tile processor
    print("🧪 Test TileProcessor...")
    
    # Mock model per test
    class MockModel(torch.nn.Module):
        def forward(self, x):
            return x * 0.9  # Simula rimozione stelle
    
    model = MockModel()
    
    processor = TileProcessor(
        model=model,
        tile_size=(256, 256),
        overlap=0.2,
        device='cpu',
        batch_size=2
    )
    
    # Test con immagine dummy
    test_img = np.random.randint(0, 255, (1000, 1500, 3), dtype=np.uint8)
    result = processor.process_image(test_img, show_progress=True)
    
    print(f"✅ Test completato!")
    print(f"  Input shape: {test_img.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Output type: {result.dtype}")