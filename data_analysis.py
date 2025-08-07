"""
Analisi del dataset per Star Removal
Analizza le immagini input e starless per comprendere le caratteristiche del dataset
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict

class DatasetAnalyzer:
    def __init__(self, input_dir, starless_dir):
        self.input_dir = Path(input_dir)
        self.starless_dir = Path(starless_dir)
        self.analysis_results = {}
        
    def analyze_images(self):
        """Analizza tutte le immagini nel dataset"""
        print("🔍 Analizzando dataset...")
        
        input_files = list(self.input_dir.glob("*.jpg"))
        starless_files = list(self.starless_dir.glob("*.jpg"))
        
        print(f"📊 Immagini input: {len(input_files)}")
        print(f"📊 Immagini starless: {len(starless_files)}")
        
        # Analizza dimensioni e caratteristiche
        input_stats = self._analyze_image_set(input_files, "Input")
        starless_stats = self._analyze_image_set(starless_files, "Starless")
        
        # Controlla corrispondenza tra input e starless
        self._check_image_pairs()
        
        # Salva risultati
        self.analysis_results = {
            "input_stats": input_stats,
            "starless_stats": starless_stats,
            "total_pairs": len(input_files),
            "image_types": self._categorize_images(input_files)
        }
        
        return self.analysis_results
    
    def _analyze_image_set(self, files, set_name):
        """Analizza un set di immagini"""
        print(f"\n📈 Analizzando set {set_name}...")
        
        stats = {
            "count": len(files),
            "dimensions": [],
            "file_sizes": [],
            "brightness": [],
            "contrast": []
        }
        
        for file_path in files[:20]:  # Analizza prime 20 per velocità
            try:
                # Leggi immagine
                img = cv2.imread(str(file_path))
                if img is None:
                    continue
                    
                h, w, c = img.shape
                stats["dimensions"].append((w, h))
                
                # Dimensione file
                stats["file_sizes"].append(file_path.stat().st_size / (1024 * 1024))  # MB
                
                # Statistiche di luminosità
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                stats["brightness"].append(np.mean(gray))
                stats["contrast"].append(np.std(gray))
                
            except Exception as e:
                print(f"Errore analizzando {file_path}: {e}")
        
        # Calcola statistiche aggregate
        if stats["dimensions"]:
            unique_dims = list(set(stats["dimensions"]))
            most_common_dim = max(set(stats["dimensions"]), key=stats["dimensions"].count)
            
            print(f"  📐 Dimensioni uniche: {len(unique_dims)}")
            print(f"  📐 Dimensione più comune: {most_common_dim}")
            print(f"  💾 Dimensione media file: {np.mean(stats['file_sizes']):.1f} MB")
            print(f"  💡 Luminosità media: {np.mean(stats['brightness']):.1f}")
            print(f"  🎨 Contrasto medio: {np.mean(stats['contrast']):.1f}")
        
        return stats
    
    def _check_image_pairs(self):
        """Controlla che ogni immagine input abbia la corrispondente starless"""
        print("\n🔍 Controllando coppie input-starless...")
        
        input_names = {f.stem for f in self.input_dir.glob("*.jpg")}
        starless_names = {f.stem for f in self.starless_dir.glob("*.jpg")}
        
        missing_starless = input_names - starless_names
        missing_input = starless_names - input_names
        matching_pairs = input_names & starless_names
        
        print(f"  ✅ Coppie corrispondenti: {len(matching_pairs)}")
        if missing_starless:
            print(f"  ❌ Input senza starless: {sorted(missing_starless)}")
        if missing_input:
            print(f"  ❌ Starless senza input: {sorted(missing_input)}")
    
    def _categorize_images(self, files):
        """Categorizza le immagini per tipo (4K vs piccole)"""
        print("\n📂 Categorizzando immagini...")
        
        categories = {
            "4k_numbers": [],  # Solo numeri, probabilmente 4K
            "small_letters": []  # Con lettere, probabilmente più piccole
        }
        
        for file_path in files:
            name = file_path.stem
            
            # Se il nome è solo numerico, probabilmente è 4K
            if name.isdigit():
                categories["4k_numbers"].append(name)
            # Se contiene lettere, probabilmente è più piccola
            elif any(c.isalpha() for c in name):
                categories["small_letters"].append(name)
        
        print(f"  🔢 Immagini 4K (numeri): {len(categories['4k_numbers'])}")
        print(f"  🔤 Immagini piccole (lettere): {len(categories['small_letters'])}")
        
        return categories
    
    def create_sample_visualization(self):
        """Crea visualizzazione di esempio delle coppie input-starless"""
        print("\n🎨 Creando visualizzazione di esempio...")
        
        # Prendi alcune coppie per la visualizzazione
        input_files = list(self.input_dir.glob("*.jpg"))[:6]
        
        fig, axes = plt.subplots(2, 6, figsize=(20, 8))
        fig.suptitle("Esempi Dataset: Input (sopra) vs Starless (sotto)", fontsize=16)
        
        for i, input_file in enumerate(input_files):
            if i >= 6:
                break
                
            starless_file = self.starless_dir / input_file.name
            
            # Carica e mostra immagine input
            img_input = cv2.imread(str(input_file))
            if img_input is not None:
                img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                axes[0, i].imshow(img_input)
                axes[0, i].set_title(f"Input: {input_file.stem}")
                axes[0, i].axis('off')
            
            # Carica e mostra immagine starless
            if starless_file.exists():
                img_starless = cv2.imread(str(starless_file))
                if img_starless is not None:
                    img_starless = cv2.cvtColor(img_starless, cv2.COLOR_BGR2RGB)
                    axes[1, i].imshow(img_starless)
                    axes[1, i].set_title(f"Starless: {starless_file.stem}")
                    axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig("star_removal_project/dataset_samples.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  ✅ Visualizzazione salvata come 'dataset_samples.png'")
    
    def save_analysis(self, output_file="dataset_analysis.json"):
        """Salva l'analisi in un file JSON"""
        output_path = f"star_removal_project/{output_file}"
        
        # Converti numpy arrays in liste per JSON
        analysis_json = {}
        for key, value in self.analysis_results.items():
            if isinstance(value, dict):
                analysis_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        if isinstance(v[0], (tuple, list, np.ndarray)):
                            analysis_json[key][k] = [list(item) if hasattr(item, '__iter__') else item for item in v]
                        else:
                            analysis_json[key][k] = list(v) if hasattr(v, '__iter__') else v
                    else:
                        analysis_json[key][k] = v
            else:
                analysis_json[key] = value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_json, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Analisi salvata in: {output_path}")

def main():
    # Path del dataset nel progetto
    input_dir = "Starlossno-1/starmodel dataset/input"
    starless_dir = "Starlossno-1/starmodel dataset/starless"
    
    analyzer = DatasetAnalyzer(input_dir, starless_dir)
    results = analyzer.analyze_images()
    analyzer.create_sample_visualization()
    analyzer.save_analysis()
    
    print("\n🎉 Analisi completata!")
    return results

if __name__ == "__main__":
    main()