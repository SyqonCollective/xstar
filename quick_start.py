"""
Quick Start Script per Star Removal Project
Avvio rapido con configurazioni ottimali per il tuo dataset
"""

import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Installa le dipendenze necessarie"""
    print("🚀 Installazione dipendenze...")
    
    try:
        # Esegui script di setup
        subprocess.run([sys.executable, "setup.py"], check=True)
        print("✅ Dipendenze installate!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore installazione: {e}")
        return False

def quick_analysis():
    """Analisi rapida del dataset"""
    print("\n🔍 Analisi rapida dataset...")
    
    try:
        from data_analysis import main as analyze_main
        analyze_main()
        print("✅ Analisi completata!")
        return True
    except Exception as e:
        print(f"❌ Errore analisi: {e}")
        return False

def quick_training():
    """Training rapido con configurazioni ottimali"""
    print("\n🚀 Avvio training rapido...")
    print("Configurazione ottimale per il tuo dataset:")
    print("  - Epochs: 50 (rapido per test)")
    print("  - Batch size: 4 (conservativo per memoria)")
    print("  - Image size: 512x512")
    print("  - Augmentation: 8x")
    print("  - Tiles: Abilitati")
    
    try:
        from main import train_model
        
        model_path = train_model(
            input_dir="../Starlossno-1/starmodel dataset/input",
            starless_dir="../Starlossno-1/starmodel dataset/starless",
            epochs=50,  # Ridotto per test rapido
            batch_size=4,  # Conservativo per memoria
            image_size=(512, 512),
            experiment_name="starnet_quickstart"
        )
        
        print(f"✅ Training completato! Modello: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Errore training: {e}")
        return None

def quick_evaluation(model_path):
    """Valutazione rapida del modello"""
    print(f"\n📊 Valutazione modello: {model_path}")
    
    try:
        from main import evaluate_model
        
        results = evaluate_model(
            model_path=model_path,
            input_dir="../Starlossno-1/starmodel dataset/input",
            starless_dir="../Starlossno-1/starmodel dataset/starless",
            output_dir="quickstart_evaluation"
        )
        
        print("\n📈 RISULTATI PRINCIPALI:")
        key_metrics = {
            'PSNR': results.get('psnr_mean', 0),
            'SSIM': results.get('ssim_mean', 0), 
            'Star Removal Rate': results.get('star_removal_rate_mean', 0),
            'Background Preservation': results.get('background_preservation_mean', 0)
        }
        
        for metric, value in key_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Errore valutazione: {e}")
        return None

def quick_processing(model_path):
    """Processamento rapido di alcune immagini"""
    print(f"\n🎨 Processamento immagini con: {model_path}")
    
    try:
        from main import process_images
        
        # Processa prime 5 immagini per test
        input_dir = "../Starlossno-1/starmodel dataset/input"
        output_dir = "quickstart_results"
        
        process_images(
            model_path=model_path,
            input_dir=input_dir,
            output_dir=output_dir,
            tile_size=(512, 512),
            overlap=0.25
        )
        
        print(f"✅ Processamento completato! Risultati in: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Errore processamento: {e}")
        return False

def main():
    """Main quick start"""
    print("🌟 STAR REMOVAL PROJECT - QUICK START")
    print("=" * 50)
    print("Questo script ti guiderà attraverso un esempio completo del progetto")
    print("con configurazioni ottimali per il tuo dataset di 63 immagini.")
    print()
    
    # Verifica se continuare
    response = input("Vuoi continuare con il quick start? (y/n): ").lower()
    if response != 'y':
        print("👋 Arrivederci!")
        return
    
    success_steps = []
    
    # Step 1: Installazione
    print("\n" + "="*50)
    print("STEP 1: INSTALLAZIONE DIPENDENZE")
    print("="*50)
    if install_dependencies():
        success_steps.append("✅ Installazione")
    else:
        print("❌ Fermata per errore installazione")
        return
    
    # Step 2: Analisi
    print("\n" + "="*50)
    print("STEP 2: ANALISI DATASET") 
    print("="*50)
    if quick_analysis():
        success_steps.append("✅ Analisi dataset")
    else:
        print("⚠️  Analisi fallita, ma continuiamo...")
    
    # Step 3: Training
    print("\n" + "="*50)
    print("STEP 3: TRAINING MODELLO")
    print("="*50)
    model_path = quick_training()
    if model_path:
        success_steps.append("✅ Training modello")
    else:
        print("❌ Training fallito, impossibile continuare")
        return
    
    # Step 4: Valutazione
    print("\n" + "="*50)
    print("STEP 4: VALUTAZIONE MODELLO")
    print("="*50)
    results = quick_evaluation(model_path)
    if results:
        success_steps.append("✅ Valutazione modello")
    
    # Step 5: Processamento
    print("\n" + "="*50)
    print("STEP 5: PROCESSAMENTO IMMAGINI")
    print("="*50)
    if quick_processing(model_path):
        success_steps.append("✅ Processamento immagini")
    
    # Summary finale
    print("\n" + "="*60)
    print("🎉 QUICK START COMPLETATO!")
    print("="*60)
    
    print("\n📋 PASSI COMPLETATI:")
    for step in success_steps:
        print(f"  {step}")
    
    print(f"\n📁 FILE IMPORTANTI:")
    print(f"  • Modello addestrato: {model_path}")
    print(f"  • Valutazione: quickstart_evaluation/")
    print(f"  • Risultati processamento: quickstart_results/")
    print(f"  • Analisi dataset: dataset_analysis.json")
    
    print(f"\n🚀 PROSSIMI PASSI:")
    print(f"  1. Esamina i risultati in quickstart_results/")
    print(f"  2. Se soddisfatto, addestra più a lungo:")
    print(f"     python main.py train --epochs 200")
    print(f"  3. Processa tutte le tue immagini:")
    print(f"     python main.py process --model-path {model_path} --input-dir /path/to/images")
    print(f"  4. Per uso avanzato, vedi main.py --help")
    
    print(f"\n💡 SUGGERIMENTI:")
    print(f"  • Per GPU più potenti, aumenta batch_size (4 -> 8 -> 16)")
    print(f"  • Per immagini più grandi, usa tile processing")
    print(f"  • Monitora training con TensorBoard")

if __name__ == "__main__":
    main()