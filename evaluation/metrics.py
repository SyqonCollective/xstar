"""
Metriche di valutazione per Star Removal Network
Include metriche specifiche per astronomia e qualità immagini
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.unet_starnet import StarNetUNet

class AstronomyMetrics:
    """
    Metriche specializzate per valutazione qualità rimozione stelle
    """
    
    @staticmethod
    def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
        """Calcola Peak Signal-to-Noise Ratio"""
        return psnr(img1, img2, data_range=max_value)
    
    @staticmethod
    def calculate_ssim(img1: np.ndarray, img2: np.ndarray, multichannel: bool = True) -> float:
        """Calcola Structural Similarity Index"""
        return ssim(img1, img2, multichannel=multichannel, data_range=255)
    
    @staticmethod
    def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcola Mean Squared Error"""
        return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    
    @staticmethod
    def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcola Mean Absolute Error"""
        return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))
    
    @staticmethod
    def star_removal_efficiency(input_img: np.ndarray, 
                               output_img: np.ndarray,
                               target_img: np.ndarray,
                               star_threshold: float = 0.8) -> Dict[str, float]:
        """
        Calcola efficienza rimozione stelle
        
        Args:
            input_img: Immagine originale con stelle
            output_img: Immagine processata
            target_img: Ground truth starless
            star_threshold: Soglia per identificare stelle
            
        Returns:
            Dictionary con metriche di rimozione stelle
        """
        
        # Converti a grayscale per analisi
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        output_gray = cv2.cvtColor(output_img, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
        
        # Identifica regioni stellari (pixel molto luminosi)
        star_mask = input_gray > (np.percentile(input_gray, 99.5))
        
        if np.sum(star_mask) == 0:
            # Nessuna stella identificata
            return {
                'star_removal_rate': 1.0,
                'star_residual': 0.0,
                'background_preservation': 1.0,
                'star_pixels_detected': 0
            }
        
        # Calcola rimozione stelle
        input_star_intensity = np.mean(input_gray[star_mask])
        output_star_intensity = np.mean(output_gray[star_mask])
        target_star_intensity = np.mean(target_gray[star_mask])
        
        # Rate di rimozione stelle
        star_removal_rate = 1.0 - (output_star_intensity - target_star_intensity) / \
                           (input_star_intensity - target_star_intensity + 1e-8)
        star_removal_rate = np.clip(star_removal_rate, 0, 1)
        
        # Residui stellari
        star_residual = np.mean(np.abs(output_gray[star_mask] - target_gray[star_mask]))
        
        # Preservazione background (regioni non stellari)
        background_mask = ~star_mask
        background_preservation = 1.0 - np.mean(np.abs(
            output_gray[background_mask] - target_gray[background_mask]
        )) / 255.0
        
        return {
            'star_removal_rate': float(star_removal_rate),
            'star_residual': float(star_residual),
            'background_preservation': float(background_preservation),
            'star_pixels_detected': int(np.sum(star_mask))
        }
    
    @staticmethod
    def calculate_gradient_preservation(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calcola quanto bene sono preservati i gradienti (dettagli fini)
        """
        # Calcola gradienti con Sobel
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        grad1_x = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        
        grad2_x = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitudine gradienti
        mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
        mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Correlazione tra gradienti
        correlation = np.corrcoef(mag1.flatten(), mag2.flatten())[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def calculate_color_preservation(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """
        Calcola preservazione dei colori
        """
        # Converti a LAB per analisi colore
        lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
        
        # Separa canali
        l1, a1, b1 = cv2.split(lab1)
        l2, a2, b2 = cv2.split(lab2)
        
        # Calcola differenze
        luminance_diff = np.mean(np.abs(l1.astype(np.float32) - l2.astype(np.float32)))
        chroma_a_diff = np.mean(np.abs(a1.astype(np.float32) - a2.astype(np.float32)))
        chroma_b_diff = np.mean(np.abs(b1.astype(np.float32) - b2.astype(np.float32)))
        
        return {
            'luminance_preservation': 1.0 - luminance_diff / 255.0,
            'chroma_a_preservation': 1.0 - chroma_a_diff / 255.0,
            'chroma_b_preservation': 1.0 - chroma_b_diff / 255.0
        }

class ModelEvaluator:
    """
    Valutatore completo per modello Star Removal
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.metrics = AstronomyMetrics()
        
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocessa immagine per il modello"""
        # Normalizza [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalizzazione ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Converti a tensor e aggiungi batch dimension
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def _postprocess_output(self, output: torch.Tensor) -> np.ndarray:
        """Postprocessa output del modello"""
        # Rimuovi batch dimension
        output = output.squeeze(0).cpu()
        
        # Denormalizza
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        output = output * std + mean
        
        # Converti a numpy e clamp
        output = output.permute(1, 2, 0).numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        
        return output
    
    def evaluate_single_image(self, 
                             input_img: np.ndarray,
                             target_img: np.ndarray) -> Dict[str, float]:
        """
        Valuta modello su singola immagine
        
        Args:
            input_img: Immagine input con stelle
            target_img: Ground truth starless
            
        Returns:
            Dictionary con tutte le metriche
        """
        
        with torch.no_grad():
            # Preprocessa e predici
            input_tensor = self._preprocess_image(input_img)
            output_tensor = self.model(input_tensor)
            output_img = self._postprocess_output(output_tensor)
        
        # Calcola metriche base
        metrics = {
            'psnr': self.metrics.calculate_psnr(output_img, target_img),
            'ssim': self.metrics.calculate_ssim(output_img, target_img),
            'mse': self.metrics.calculate_mse(output_img, target_img),
            'mae': self.metrics.calculate_mae(output_img, target_img)
        }
        
        # Metriche di rimozione stelle
        star_metrics = self.metrics.star_removal_efficiency(
            input_img, output_img, target_img
        )
        metrics.update(star_metrics)
        
        # Preservazione gradienti
        metrics['gradient_preservation'] = self.metrics.calculate_gradient_preservation(
            target_img, output_img
        )
        
        # Preservazione colori
        color_metrics = self.metrics.calculate_color_preservation(target_img, output_img)
        metrics.update(color_metrics)
        
        return metrics, output_img
    
    def evaluate_dataset(self,
                        input_dir: str,
                        starless_dir: str,
                        output_dir: Optional[str] = None,
                        save_predictions: bool = True,
                        max_images: Optional[int] = None) -> Dict[str, float]:
        """
        Valuta modello su intero dataset
        
        Args:
            input_dir: Directory immagini input
            starless_dir: Directory immagini starless
            output_dir: Directory per salvare predizioni (opzionale)
            save_predictions: Se salvare le predizioni
            max_images: Numero massimo di immagini da valutare
            
        Returns:
            Dictionary con metriche aggregate
        """
        
        input_path = Path(input_dir)
        starless_path = Path(starless_dir)
        
        if output_dir and save_predictions:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Trova immagini corrispondenti
        input_files = list(input_path.glob("*.jpg"))
        if max_images:
            input_files = input_files[:max_images]
        
        # Accumula metriche
        all_metrics = []
        
        print(f"🔍 Valutando {len(input_files)} immagini...")
        
        for input_file in tqdm(input_files, desc="Valutazione"):
            starless_file = starless_path / input_file.name
            
            if not starless_file.exists():
                print(f"⚠️  Starless non trovato per {input_file.name}")
                continue
            
            try:
                # Carica immagini
                input_img = cv2.imread(str(input_file))
                starless_img = cv2.imread(str(starless_file))
                
                if input_img is None or starless_img is None:
                    continue
                
                # Converti BGR -> RGB
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                starless_img = cv2.cvtColor(starless_img, cv2.COLOR_BGR2RGB)
                
                # Valuta
                metrics, prediction = self.evaluate_single_image(input_img, starless_img)
                metrics['filename'] = input_file.stem
                all_metrics.append(metrics)
                
                # Salva predizione se richiesto
                if output_dir and save_predictions:
                    pred_file = output_path / f"pred_{input_file.name}"
                    pred_bgr = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(pred_file), pred_bgr)
                
            except Exception as e:
                print(f"❌ Errore valutando {input_file.name}: {e}")
                continue
        
        if not all_metrics:
            raise ValueError("Nessuna immagine valutata con successo")
        
        # Calcola statistiche aggregate
        aggregate_metrics = {}
        metric_keys = [k for k in all_metrics[0].keys() if k != 'filename']
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregate_metrics[f"{key}_mean"] = np.mean(values)
                aggregate_metrics[f"{key}_std"] = np.std(values)
                aggregate_metrics[f"{key}_min"] = np.min(values)
                aggregate_metrics[f"{key}_max"] = np.max(values)
        
        # Salva risultati dettagliati
        if output_dir:
            results_file = output_path / "evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'aggregate_metrics': aggregate_metrics,
                    'individual_metrics': all_metrics
                }, f, indent=2)
            
            print(f"💾 Risultati salvati in: {results_file}")
        
        return aggregate_metrics
    
    def create_evaluation_report(self,
                               results: Dict[str, float],
                               output_path: str):
        """Crea report di valutazione con visualizzazioni"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Star Removal Model - Evaluation Report', fontsize=16)
        
        # Metriche principali
        main_metrics = ['psnr_mean', 'ssim_mean', 'star_removal_rate_mean']
        main_values = [results.get(m, 0) for m in main_metrics]
        main_labels = ['PSNR', 'SSIM', 'Star Removal Rate']
        
        axes[0, 0].bar(main_labels, main_values)
        axes[0, 0].set_title('Main Quality Metrics')
        axes[0, 0].set_ylim(0, 1)
        
        # Distribuzione errori
        error_metrics = ['mse_mean', 'mae_mean']
        error_values = [results.get(m, 0) for m in error_metrics]
        error_labels = ['MSE', 'MAE']
        
        axes[0, 1].bar(error_labels, error_values)
        axes[0, 1].set_title('Error Metrics')
        
        # Preservazione caratteristiche
        preservation_metrics = ['background_preservation_mean', 'gradient_preservation_mean']
        preservation_values = [results.get(m, 0) for m in preservation_metrics]
        preservation_labels = ['Background', 'Gradients']
        
        axes[0, 2].bar(preservation_labels, preservation_values)
        axes[0, 2].set_title('Feature Preservation')
        axes[0, 2].set_ylim(0, 1)
        
        # Colori
        color_metrics = ['luminance_preservation_mean', 'chroma_a_preservation_mean', 'chroma_b_preservation_mean']
        color_values = [results.get(m, 0) for m in color_metrics]
        color_labels = ['Luminance', 'Chroma A', 'Chroma B']
        
        axes[1, 0].bar(color_labels, color_values)
        axes[1, 0].set_title('Color Preservation')
        axes[1, 0].set_ylim(0, 1)
        
        # Statistiche variabilità
        variability_metrics = ['psnr_std', 'ssim_std', 'star_removal_rate_std']
        variability_values = [results.get(m, 0) for m in variability_metrics]
        variability_labels = ['PSNR', 'SSIM', 'Star Removal']
        
        axes[1, 1].bar(variability_labels, variability_values)
        axes[1, 1].set_title('Metric Variability (Std Dev)')
        
        # Summary text
        axes[1, 2].axis('off')
        summary_text = f"""
Model Performance Summary:

🎯 Overall Quality:
  • PSNR: {results.get('psnr_mean', 0):.2f} ± {results.get('psnr_std', 0):.2f}
  • SSIM: {results.get('ssim_mean', 0):.3f} ± {results.get('ssim_std', 0):.3f}

⭐ Star Removal:
  • Removal Rate: {results.get('star_removal_rate_mean', 0):.3f}
  • Background Preservation: {results.get('background_preservation_mean', 0):.3f}

🎨 Feature Preservation:
  • Gradients: {results.get('gradient_preservation_mean', 0):.3f}
  • Colors: {results.get('luminance_preservation_mean', 0):.3f}
        """
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Report salvato: {output_path}")

def evaluate_model_from_checkpoint(checkpoint_path: str,
                                 input_dir: str,
                                 starless_dir: str,
                                 output_dir: str = "evaluation_results"):
    """
    Funzione helper per valutare modello da checkpoint
    """
    
    # Carica modello
    model = StarNetUNet(n_channels=3, n_classes=3)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Crea evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = ModelEvaluator(model, device=device)
    
    # Valuta
    results = evaluator.evaluate_dataset(
        input_dir=input_dir,
        starless_dir=starless_dir,
        output_dir=output_dir,
        save_predictions=True
    )
    
    # Crea report
    report_path = Path(output_dir) / "evaluation_report.png"
    evaluator.create_evaluation_report(results, str(report_path))
    
    return results

if __name__ == "__main__":
    # Test delle metriche
    print("🧪 Test metriche...")
    
    # Crea immagini di test
    input_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    target_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    metrics = AstronomyMetrics()
    
    # Test metriche base
    psnr_val = metrics.calculate_psnr(input_img, target_img)
    ssim_val = metrics.calculate_ssim(input_img, target_img)
    
    print(f"PSNR: {psnr_val:.2f}")
    print(f"SSIM: {ssim_val:.3f}")
    
    # Test metriche stellari
    star_metrics = metrics.star_removal_efficiency(input_img, target_img, target_img)
    print(f"Star metrics: {star_metrics}")
    
    print("✅ Test completato!")