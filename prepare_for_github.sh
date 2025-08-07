#!/bin/bash
# Script di preparazione per GitHub
# Pulisce il progetto e verifica che tutto sia pronto per il push

echo "🚀 Preparazione Star Removal Project per GitHub..."

# Colori per output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Funzione per stampe colorate
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 1. Verifica struttura progetto
echo "📁 Verifica struttura progetto..."
required_files=(
    "README.md"
    "requirements.txt"
    "setup.py"
    "main.py"
    "quick_start.py"
    ".gitignore"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_success "Trovato: $file"
    else
        print_error "Mancante: $file"
        exit 1
    fi
done

# 2. Verifica dataset
echo ""
echo "🖼️  Verifica dataset..."
if [ -d "Starlossno-1" ]; then
    input_count=$(find Starlossno-1/starmodel\ dataset/input -name "*.jpg" 2>/dev/null | wc -l)
    starless_count=$(find Starlossno-1/starmodel\ dataset/starless -name "*.jpg" 2>/dev/null | wc -l)
    
    echo "   Input images: $input_count"
    echo "   Starless images: $starless_count"
    
    if [ $input_count -gt 0 ] && [ $starless_count -gt 0 ]; then
        print_success "Dataset trovato con $input_count immagini"
    else
        print_warning "Dataset sembra vuoto o incompleto"
    fi
else
    print_warning "Directory dataset non trovata"
fi

# 3. Pulisce file temporanei e checkpoint
echo ""
echo "🧹 Pulizia file temporanei..."

# Rimuovi checkpoint (esclusi da .gitignore ma puliamo comunque)
if [ -d "outputs" ]; then
    echo "   Rimozione checkpoint da outputs/..."
    find outputs -name "*.pth" -delete 2>/dev/null
    find outputs -name "checkpoint_*" -delete 2>/dev/null
    print_success "Checkpoint rimossi"
fi

# Rimuovi cache Python
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
print_success "Cache Python pulita"

# Rimuovi file temporanei
find . -name "*.tmp" -delete 2>/dev/null
find . -name "*.temp" -delete 2>/dev/null
find . -name ".DS_Store" -delete 2>/dev/null
print_success "File temporanei rimossi"

# 4. Verifica Git
echo ""
echo "📦 Verifica Git..."
if [ -d ".git" ]; then
    print_success "Repository Git già inizializzato"
else
    echo "   Inizializzazione repository Git..."
    git init
    git add .gitignore
    print_success "Repository Git inizializzato"
fi

# 5. Calcola dimensioni
echo ""
echo "📊 Calcolo dimensioni progetto..."

# Dimensione totale
total_size=$(du -sh . | cut -f1)
echo "   Dimensione totale: $total_size"

# Dimensione senza dataset (per vedere quanto spazio occupa il codice)
code_size=$(du -sh --exclude="Starlossno-1" . | cut -f1)
echo "   Dimensione codice: $code_size"

# Dimensione dataset
if [ -d "Starlossno-1" ]; then
    dataset_size=$(du -sh "Starlossno-1" | cut -f1)
    echo "   Dimensione dataset: $dataset_size"
fi

# 6. Verifica dependenze Python
echo ""
echo "🐍 Verifica dipendenze Python..."
if python3 -c "import sys; print(f'Python {sys.version}')" 2>/dev/null; then
    print_success "Python disponibile"
    
    # Testa import principali
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        print_success "PyTorch disponibile"
    else
        print_warning "PyTorch non installato (OK per GitHub)"
    fi
else
    print_error "Python non disponibile"
fi

# 7. Crea README per GitHub
echo ""
echo "📝 Aggiornamento informazioni GitHub..."

# Verifica se il README ha le informazioni per GitHub
if grep -q "GitHub Actions" README.md; then
    print_success "README già configurato per GitHub"
else
    print_warning "Considera di aggiungere informazioni su GitHub Actions al README"
fi

# 8. Suggerimenti finali
echo ""
echo "🎯 Preparazione completata!"
echo ""
echo "📋 Prossimi passi per GitHub:"
echo "   1. git add ."
echo "   2. git commit -m 'Initial commit: Star Removal Project'"
echo "   3. git branch -M main"
echo "   4. git remote add origin https://github.com/TUO_USERNAME/star-removal-project.git"
echo "   5. git push -u origin main"
echo ""
echo "🚀 Per avviare training su GitHub Actions:"
echo "   • Vai su Actions tab nel repository"
echo "   • Seleziona 'Star Removal Training'"
echo "   • Click 'Run workflow'"
echo "   • Configura epochs, batch_size, etc."
echo ""

# 9. Warning su dimensioni
if [ -d "Starlossno-1" ]; then
    dataset_size_mb=$(du -sm "Starlossno-1" | cut -f1)
    if [ $dataset_size_mb -gt 100 ]; then
        print_warning "Dataset > 100MB - considera Git LFS:"
        echo "   git lfs track \"*.jpg\""
        echo "   git lfs track \"*.png\""
        echo "   git add .gitattributes"
    fi
fi

echo ""
print_success "Progetto pronto per GitHub! 🌟"
