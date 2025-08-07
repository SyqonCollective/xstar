#!/usr/bin/env python3
"""
Quick Star Removal - Processamento immediato senza training
Usa algoritmi classici per rimozione stelle rapida
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def quick_star_removal(image_path, output_path):
    """
    Rimozione stelle veloce con algoritmi classici
    """
    print(f"🚀 Processando: {image_path}")
    
    # Carica immagine
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Errore caricamento: {image_path}")
        return
    
    # Converti a RGB per processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Metodo 1: Morphological Opening per stelle piccole
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Identifica stelle (punti molto luminosi)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Morfologia per trovare stelle puntiformi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stars_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Dilata la maschera per coprire meglio le stelle
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    stars_mask = cv2.dilate(stars_mask, kernel_dilate, iterations=1)
    
    # Inpainting per rimuovere stelle
    result = cv2.inpaint(img, stars_mask, 3, cv2.INPAINT_TELEA)
    
    # Salva risultato
    cv2.imwrite(str(output_path), result)
    print(f"✅ Salvato: {output_path}")
    
    return result

def process_image(input_path, output_dir="results"):
    """Processa singola immagine"""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"{input_path.stem}_starless{input_path.suffix}"
    
    quick_star_removal(input_path, output_path)
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Star Removal")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    result = process_image(args.input, args.output_dir)
    print(f"🌟 Processing completato: {result}")