"""
Script principale per Star Removal Project
Orchestrazione completa: analisi dati, training, valutazione e inferenza
"""

import argparse
import sys
from pathlib import Path
import torch
import json

# Aggiungi il progetto al path
sys.path.append(str(Path(__file__).parent))

try:
    from data_analysis import DatasetAnalyzer
    from training.trainer import create_trainer
    from inference.tile_processor import TileProcessor
    from evaluation.metrics import evaluate_model_from_checkpoint
    from models.unet_starnet import StarNetUNet
except ImportError:
    # Aggiungi il path del progetto se necessario
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    from data_analysis import DatasetAnalyzer
    from training.trainer import create_trainer
    from inference.tile_processor import TileProcessor
    from evaluation.metrics import evaluate_model_from_checkpoint
    from models.unet_starnet import StarNetUNet

def analyze_dataset(input_dir: str, starless_dir: str):
    """Analizza il dataset"""
    print("🔍 FASE 1: Analisi Dataset")
    print("=" * 50)
    
    analyzer = DatasetAnalyzer(input_dir, starless_dir)
    results = analyzer.analyze_images()
    analyzer.create_sample_visualization()
    analyzer.save_analysis()
    
    print("\n✅ Analisi completata!")
    return results

def train_model(input_dir: str,
               starless_dir: str,
               epochs: int = 100,
               batch_size: int = 8,
               image_size: tuple = (512, 512),
               num_workers: int = 4,
               experiment_name: str = "starnet_v1"):
    """Addestra il modello"""
    print(f"\n🚀 FASE 2: Training Modello")
    print("=" * 50)
    
    # Crea trainer
    trainer = create_trainer(
        input_dir=input_dir,
        starless_dir=starless_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        experiment_name=experiment_name
    )    # Addestra
    trainer.train(num_epochs=epochs, save_every=10)
    
    # Restituisci path del miglior modello
    best_model_path = trainer.output_dir / "best_model.pth"
    
    print(f"\n✅ Training completato!")
    print(f"📁 Modello salvato: {best_model_path}")
    
    return str(best_model_path)

def evaluate_model(model_path: str,
                  input_dir: str,
                  starless_dir: str,
                  output_dir: str = "evaluation_results"):
    """Valuta il modello"""
    print(f"\n📊 FASE 3: Valutazione Modello")
    print("=" * 50)
    
    results = evaluate_model_from_checkpoint(
        checkpoint_path=model_path,
        input_dir=input_dir,
        starless_dir=starless_dir,
        output_dir=output_dir
    )
    
    print("\n✅ Valutazione completata!")
    print(f"📁 Risultati in: {output_dir}")
    
    return results

def process_images(model_path: str,
                  input_dir: str,
                  output_dir: str,
                  tile_size: tuple = (512, 512),
                  overlap: float = 0.25):
    """Processa immagini con il modello addestrato"""
    print(f"\n🎨 FASE 4: Processamento Immagini")
    print("=" * 50)
    
    # Carica modello
    model = StarNetUNet(n_channels=3, n_classes=3)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Crea processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = TileProcessor(
        model=model,
        tile_size=tile_size,
        overlap=overlap,
        device=device,
        batch_size=4
    )
    
    # Processa directory
    processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern="*.jpg"
    )
    
    print(f"\n✅ Processamento completato!")
    print(f"📁 Immagini processate in: {output_dir}")

def setup_project():
    """Setup iniziale del progetto"""
    print("🛠️  SETUP: Configurazione Progetto")
    print("=" * 50)
    
    # Crea struttura directory
    directories = [
        "star_removal_project/data",
        "star_removal_project/models",
        "star_removal_project/training", 
        "star_removal_project/inference",
        "star_removal_project/evaluation",
        "star_removal_project/outputs",
        "star_removal_project/results"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Crea __init__.py files
    init_files = [
        "star_removal_project/__init__.py",
        "star_removal_project/data/__init__.py",
        "star_removal_project/models/__init__.py",
        "star_removal_project/training/__init__.py",
        "star_removal_project/inference/__init__.py",
        "star_removal_project/evaluation/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    print("✅ Struttura progetto creata!")

def main():
    parser = argparse.ArgumentParser(description="Star Removal Project - Training e Inferenza")
    
    # Comando principale
    parser.add_argument("command", choices=["setup", "analyze", "train", "evaluate", "process", "full"],
                       help="Comando da eseguire")
    
    # Directory del dataset
    parser.add_argument("--input-dir", default="Starlossno-1/starmodel dataset/input",
                       help="Directory immagini input")
    parser.add_argument("--starless-dir", default="Starlossno-1/starmodel dataset/starless", 
                       help="Directory immagini starless")
    
    # Parametri training
    parser.add_argument("--epochs", type=int, default=100,
                       help="Numero di epochs per training")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size per training")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512],
                       help="Dimensioni immagini per training")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Numero di worker per DataLoader")
    parser.add_argument("--experiment-name", default="starnet_v1",
                       help="Nome esperimento")
    
    # Parametri inferenza
    parser.add_argument("--model-path", 
                       help="Path del modello addestrato")
    parser.add_argument("--output-dir", default="results",
                       help="Directory output")
    parser.add_argument("--tile-size", type=int, nargs=2, default=[512, 512],
                       help="Dimensioni tile per inferenza")
    parser.add_argument("--overlap", type=float, default=0.25,
                       help="Overlap percentuale tra tile")
    
    args = parser.parse_args()
    
    # Esegui comando
    try:
        if args.command == "setup":
            setup_project()
            
        elif args.command == "analyze":
            analyze_dataset(args.input_dir, args.starless_dir)
            
        elif args.command == "train":
            model_path = train_model(
                input_dir=args.input_dir,
                starless_dir=args.starless_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=tuple(args.image_size),
                num_workers=args.num_workers,
                experiment_name=args.experiment_name
            )
            print(f"\n🎯 Modello addestrato: {model_path}")
            
        elif args.command == "evaluate":
            if not args.model_path:
                print("❌ Specifica --model-path per la valutazione")
                return
            
            results = evaluate_model(
                model_path=args.model_path,
                input_dir=args.input_dir,
                starless_dir=args.starless_dir,
                output_dir=args.output_dir
            )
            
            # Stampa risultati chiave
            print("\n📊 RISULTATI CHIAVE:")
            key_metrics = ['psnr_mean', 'ssim_mean', 'star_removal_rate_mean']
            for metric in key_metrics:
                if metric in results:
                    print(f"  {metric}: {results[metric]:.4f}")
            
        elif args.command == "process":
            if not args.model_path:
                print("❌ Specifica --model-path per il processamento")
                return
            
            process_images(
                model_path=args.model_path,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                tile_size=tuple(args.tile_size),
                overlap=args.overlap
            )
            
        elif args.command == "full":
            # Pipeline completa
            print("🚀 PIPELINE COMPLETA - STAR REMOVAL PROJECT")
            print("=" * 60)
            
            # 1. Setup
            setup_project()
            
            # 2. Analisi
            analyze_dataset(args.input_dir, args.starless_dir)
            
            # 3. Training
            model_path = train_model(
                input_dir=args.input_dir,
                starless_dir=args.starless_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=tuple(args.image_size),
                experiment_name=args.experiment_name
            )
            
            # 4. Valutazione
            evaluation_dir = f"evaluation_{args.experiment_name}"
            results = evaluate_model(
                model_path=model_path,
                input_dir=args.input_dir,
                starless_dir=args.starless_dir,
                output_dir=evaluation_dir
            )
            
            # 5. Processamento finale
            final_output_dir = f"final_results_{args.experiment_name}"
            process_images(
                model_path=model_path,
                input_dir=args.input_dir,
                output_dir=final_output_dir,
                tile_size=tuple(args.tile_size),
                overlap=args.overlap
            )
            
            print("\n🎉 PIPELINE COMPLETA TERMINATA!")
            print(f"📁 Modello: {model_path}")
            print(f"📁 Valutazione: {evaluation_dir}")
            print(f"📁 Risultati finali: {final_output_dir}")
            
            # Salva summary
            summary = {
                "model_path": model_path,
                "evaluation_dir": evaluation_dir,
                "final_output_dir": final_output_dir,
                "key_metrics": {k: v for k, v in results.items() 
                              if any(metric in k for metric in ['psnr_mean', 'ssim_mean', 'star_removal_rate_mean'])}
            }
            
            summary_file = f"project_summary_{args.experiment_name}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📄 Summary salvato: {summary_file}")
        
    except Exception as e:
        print(f"❌ Errore durante esecuzione: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())