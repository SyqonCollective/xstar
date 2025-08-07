"""
Star Removal Project
Progetto completo per rimozione stelle da immagini astronomiche usando U-Net
"""

__version__ = "1.0.0"
__author__ = "Star Removal Team"

from .models.unet_starnet import StarNetUNet, StarNetLoss
from .training.trainer import StarRemovalTrainer
from .inference.tile_processor import TileProcessor
from .evaluation.metrics import ModelEvaluator, AstronomyMetrics

__all__ = [
    'StarNetUNet',
    'StarNetLoss', 
    'StarRemovalTrainer',
    'TileProcessor',
    'ModelEvaluator',
    'AstronomyMetrics'
]