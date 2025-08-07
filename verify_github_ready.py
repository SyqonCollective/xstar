#!/usr/bin/env python3
"""
Script di verifica pre-GitHub per Star Removal Project
Controlla che tutto sia pronto per il push su GitHub
"""

import os
import sys
from pathlib import Path
import subprocess
import json

def print_success(msg):
    print(f"✅ {msg}")

def print_warning(msg):
    print(f"⚠️  {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def print_info(msg):
    print(f"ℹ️  {msg}")

def check_file_exists(file_path):
    """Verifica che un file esista"""
    return Path(file_path).exists()

def get_file_size(file_path):
    """Ottiene la dimensione di un file in MB"""
    try:
        size_bytes = Path(file_path).stat().st_size
        return size_bytes / (1024 * 1024)  # Convert to MB
    except:
        return 0

def count_images_in_dir(directory):
    """Conta le immagini in una directory"""
    if not Path(directory).exists():
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    count = 0
    
    for ext in image_extensions:
        count += len(list(Path(directory).rglob(f"*{ext}")))
    
    return count

def main():
    print("🚀 Verifica Star Removal Project per GitHub")
    print("=" * 50)
    
    # 1. Verifica file essenziali
    print("\n📁 Verifica file essenziali...")
    essential_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'main.py',
        'quick_start.py',
        '.gitignore',
        '.gitattributes',
        'Dockerfile',
        'docker-compose.yml'
    ]
    
    missing_files = []
    for file in essential_files:
        if check_file_exists(file):
            print_success(f"Trovato: {file}")
        else:
            print_error(f"Mancante: {file}")
            missing_files.append(file)
    
    if missing_files:
        print_error(f"File mancanti: {', '.join(missing_files)}")
        return False
    
    # 2. Verifica GitHub Actions
    print("\n🔄 Verifica GitHub Actions...")
    github_workflows = Path('.github/workflows')
    if github_workflows.exists():
        workflow_files = list(github_workflows.glob('*.yml'))
        if workflow_files:
            print_success(f"Trovati {len(workflow_files)} workflow GitHub Actions")
            for wf in workflow_files:
                print_info(f"  - {wf.name}")
        else:
            print_warning("Nessun workflow GitHub Actions trovato")
    else:
        print_warning("Directory .github/workflows non trovata")
    
    # 3. Verifica dataset
    print("\n🖼️  Verifica dataset...")
    dataset_dir = Path("Starlossno-1/starmodel dataset")
    if dataset_dir.exists():
        input_dir = dataset_dir / "input"
        starless_dir = dataset_dir / "starless"
        
        input_count = count_images_in_dir(input_dir)
        starless_count = count_images_in_dir(starless_dir)
        
        print_info(f"Immagini input: {input_count}")
        print_info(f"Immagini starless: {starless_count}")
        
        if input_count > 0 and starless_count > 0:
            print_success(f"Dataset OK: {input_count} coppie di immagini")
        else:
            print_warning("Dataset sembra incompleto")
        
        # Calcola dimensione dataset
        try:
            result = subprocess.run(['du', '-sm', 'Starlossno-1'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                size_mb = int(result.stdout.split()[0])
                print_info(f"Dimensione dataset: {size_mb} MB")
                
                if size_mb > 100:
                    print_warning("Dataset > 100MB - Git LFS raccomandato")
                    # Verifica Git LFS setup
                    if check_file_exists('.gitattributes'):
                        print_success("File .gitattributes trovato per Git LFS")
                    else:
                        print_warning("File .gitattributes mancante per Git LFS")
        except:
            pass
    else:
        print_warning("Dataset directory non trovata")
    
    # 4. Verifica struttura moduli Python
    print("\n🐍 Verifica struttura Python...")
    python_modules = ['data', 'models', 'training', 'evaluation', 'inference']
    
    for module in python_modules:
        module_dir = Path(module)
        if module_dir.exists():
            init_file = module_dir / '__init__.py'
            if init_file.exists():
                print_success(f"Modulo {module}: OK")
            else:
                print_warning(f"Modulo {module}: __init__.py mancante")
        else:
            print_error(f"Modulo {module}: directory mancante")
    
    # 5. Verifica che non ci siano checkpoint
    print("\n🧹 Verifica pulizia...")
    checkpoint_files = list(Path('.').rglob('*.pth')) + list(Path('.').rglob('checkpoint_*'))
    
    if checkpoint_files:
        print_warning(f"Trovati {len(checkpoint_files)} file checkpoint:")
        for cp in checkpoint_files[:5]:  # Mostra solo i primi 5
            print_info(f"  - {cp}")
        if len(checkpoint_files) > 5:
            print_info(f"  ... e altri {len(checkpoint_files) - 5}")
        print_info("Questi file saranno ignorati da Git (.gitignore)")
    else:
        print_success("Nessun file checkpoint trovato")
    
    # 6. Verifica dipendenze
    print("\n📦 Verifica requirements.txt...")
    if check_file_exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read().splitlines()
        
        # Filtra commenti e righe vuote
        reqs = [line.strip() for line in requirements 
                if line.strip() and not line.strip().startswith('#')]
        
        print_success(f"Trovate {len(reqs)} dipendenze")
        
        # Verifica dipendenze critiche
        critical_deps = ['torch', 'torchvision', 'numpy', 'opencv-python']
        for dep in critical_deps:
            if any(dep in req for req in reqs):
                print_success(f"  ✓ {dep}")
            else:
                print_warning(f"  ? {dep} non trovato esplicitamente")
    
    # 7. Stima risorse GitHub Actions
    print("\n🔋 Stima risorse GitHub Actions...")
    print_info("GitHub Actions (free tier):")
    print_info("  - 2000 minuti/mese")
    print_info("  - 2 CPU cores")
    print_info("  - 7GB RAM")
    print_info("  - 14GB SSD storage")
    print_info("")
    print_info("Per questo progetto:")
    print_info("  - Training stimato: 30-60 min (CPU)")
    print_info("  - Memoria utilizzata: ~2-4GB")
    print_info("  - Storage necessario: ~1-5GB")
    
    # 8. Suggerimenti finali
    print("\n🎯 Suggerimenti finali:")
    print("1. Esegui: ./prepare_for_github.sh")
    print("2. git lfs install && git lfs track '*.jpg'")
    print("3. git add . && git commit -m 'Initial commit'")
    print("4. git remote add origin <your-repo-url>")
    print("5. git push -u origin main")
    print("\n🚀 Il progetto sarà pronto per il training su GitHub!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
