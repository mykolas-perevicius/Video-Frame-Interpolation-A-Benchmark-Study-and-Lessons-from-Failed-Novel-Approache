#!/bin/bash
# setup_wsl.sh - WSL2-optimized setup for Gaming VFI+SR Project
#
# Usage:
#   chmod +x setup_wsl.sh
#   ./setup_wsl.sh
#
# This script:
# 1. Installs system dependencies
# 2. Creates Python virtual environment
# 3. Installs PyTorch with CUDA
# 4. Clones model repositories
# 5. Downloads pretrained weights
# 6. Creates project structure

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_err() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "======================================"
echo " Gaming VFI+SR Setup (WSL2)"
echo "======================================"
echo ""

# Check if running in WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo_info "WSL detected"
else
    echo_warn "Not running in WSL - script may need modifications"
fi

# Check NVIDIA GPU access
echo ""
echo_info "Checking GPU access..."
if ! command -v nvidia-smi &> /dev/null; then
    echo_err "nvidia-smi not found!"
    echo "Make sure you have NVIDIA drivers installed on Windows (not in WSL)"
    echo "Required: NVIDIA Driver 470.76+ (recommend 535+)"
    exit 1
fi

echo_info "GPU detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Project directory - use WSL filesystem for speed
PROJECT_DIR="$HOME/gaming-vfisr"
echo_info "Project directory: $PROJECT_DIR"
echo_warn "IMPORTANT: Always work in WSL filesystem (~/), NOT /mnt/c/"
echo ""

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# ============================================================
# STAGE 1: System Dependencies
# ============================================================
echo ""
echo "======================================"
echo " [1/8] System Dependencies"
echo "======================================"

echo_info "Updating package lists..."
sudo apt-get update -qq

echo_info "Installing system packages..."
sudo apt-get install -y \
    git git-lfs \
    ffmpeg \
    python3.10 python3.10-venv python3-pip python3.10-dev \
    build-essential cmake \
    libopencv-dev \
    unzip wget curl \
    htop nvtop 2>/dev/null || sudo apt-get install -y htop

# Note: We do NOT install nvidia-cuda-toolkit in WSL2
# CUDA comes from Windows driver automatically

# ============================================================
# STAGE 2: Python Virtual Environment
# ============================================================
echo ""
echo "======================================"
echo " [2/8] Python Virtual Environment"
echo "======================================"

if [ -d "venv" ]; then
    echo_warn "Virtual environment already exists, skipping creation"
else
    echo_info "Creating virtual environment with Python 3.10..."
    python3.10 -m venv venv
fi

echo_info "Activating virtual environment..."
source venv/bin/activate

echo_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools -q

# ============================================================
# STAGE 3: PyTorch with CUDA
# ============================================================
echo ""
echo "======================================"
echo " [3/8] PyTorch with CUDA"
echo "======================================"

echo_info "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

echo_info "Verifying CUDA..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  ERROR: CUDA not available!')
    exit(1)
"

# ============================================================
# STAGE 4: Core Dependencies
# ============================================================
echo ""
echo "======================================"
echo " [4/8] Core Dependencies"
echo "======================================"

echo_info "Installing core packages..."
pip install -q \
    numpy pandas scipy \
    opencv-python-headless av decord ffmpeg-python \
    matplotlib seaborn plotly \
    hydra-core omegaconf \
    wandb \
    tqdm rich \
    pytest \
    scikit-image scikit-learn \
    Pillow

# ============================================================
# STAGE 5: ML/Metrics Packages
# ============================================================
echo ""
echo "======================================"
echo " [5/8] ML and Metrics Packages"
echo "======================================"

echo_info "Installing ML packages..."
pip install -q \
    pyiqa \
    lpips \
    einops \
    timm \
    transformers \
    diffusers accelerate

# TensorRT is tricky in WSL - skip for now
echo_warn "Skipping TensorRT (can be added later for optimization)"

# ============================================================
# STAGE 6: Clone Model Repositories
# ============================================================
echo ""
echo "======================================"
echo " [6/8] Clone Model Repositories"
echo "======================================"

mkdir -p external
cd external

# RIFE v4.25/4.26 - Speed champion
if [ ! -d "Practical-RIFE" ]; then
    echo_info "Cloning RIFE..."
    git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git
else
    echo_info "RIFE already cloned"
fi

# VFIMamba - Quality SOTA
if [ ! -d "VFIMamba" ]; then
    echo_info "Cloning VFIMamba..."
    git clone --depth 1 https://github.com/MCG-NJU/VFIMamba.git
else
    echo_info "VFIMamba already cloned"
fi

# SAFA - Joint VFI+SR SOTA
if [ ! -d "WACV2024-SAFA" ]; then
    echo_info "Cloning SAFA..."
    git clone --depth 1 https://github.com/hzwer/WACV2024-SAFA.git
else
    echo_info "SAFA already cloned"
fi

# SPAN - SR speed winner
if [ ! -d "SPAN" ]; then
    echo_info "Cloning SPAN..."
    git clone --depth 1 https://github.com/hongyuanyu/SPAN.git
else
    echo_info "SPAN already cloned"
fi

# EDEN - CVPR 2025 diffusion VFI (optional, slower)
if [ ! -d "EDEN" ]; then
    echo_info "Cloning EDEN..."
    git clone --depth 1 https://github.com/bbldCVer/EDEN.git 2>/dev/null || echo_warn "EDEN clone failed - may not be available yet"
else
    echo_info "EDEN already cloned"
fi

cd "$PROJECT_DIR"

# ============================================================
# STAGE 7: Download Pretrained Weights
# ============================================================
echo ""
echo "======================================"
echo " [7/8] Download Pretrained Weights"
echo "======================================"

# RIFE weights
cd external/Practical-RIFE
if [ ! -d "train_log" ]; then
    echo_info "Downloading RIFE v4.25 weights..."
    wget -q --show-progress https://github.com/hzwer/Practical-RIFE/releases/download/v4.25/train_log.zip
    unzip -q train_log.zip
    rm train_log.zip
    echo_info "RIFE weights downloaded"
else
    echo_info "RIFE weights already present"
fi
cd "$PROJECT_DIR"

# VFIMamba weights - needs manual download
echo_warn "VFIMamba weights need manual download from:"
echo "  https://github.com/MCG-NJU/VFIMamba#pretrained-models"
echo "  Place in: external/VFIMamba/checkpoints/"

# SPAN weights - needs manual download
echo_warn "SPAN weights may need manual download from:"
echo "  https://github.com/hongyuanyu/SPAN#pretrained-models"

# ============================================================
# STAGE 8: Create Project Structure
# ============================================================
echo ""
echo "======================================"
echo " [8/8] Create Project Structure"
echo "======================================"

echo_info "Creating directory structure..."

mkdir -p config/models config/experiments
mkdir -p scripts
mkdir -p models/{traditional,sota,novel}
mkdir -p data/{raw,processed}
mkdir -p evaluation
mkdir -p outputs/{benchmarks,visualizations,blind_study,model_outputs}
mkdir -p notebooks
mkdir -p tests

# Create __init__.py files
touch models/__init__.py
touch models/traditional/__init__.py
touch models/sota/__init__.py
touch models/novel/__init__.py
touch evaluation/__init__.py

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Video I/O
opencv-python-headless>=4.8.0
av>=10.0.0
decord>=0.6.0
ffmpeg-python>=0.2.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Configuration
hydra-core>=1.3.0
omegaconf>=2.3.0

# Experiment tracking
wandb>=0.15.0

# Progress and CLI
tqdm>=4.65.0
rich>=13.0.0

# Testing
pytest>=7.3.0

# Image processing
scikit-image>=0.21.0
scikit-learn>=1.3.0
Pillow>=10.0.0

# Quality metrics
pyiqa>=0.1.7
lpips>=0.1.4

# Deep learning utilities
einops>=0.6.0
timm>=0.9.0
transformers>=4.30.0
diffusers>=0.20.0
accelerate>=0.20.0
EOF

echo_info "Created requirements.txt"

# Create activation script
cat > activate.sh << 'EOF'
#!/bin/bash
# Quick activation script
cd ~/gaming-vfisr
source venv/bin/activate
echo "Environment activated!"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not found')"
EOF
chmod +x activate.sh

echo_info "Created activate.sh"

# ============================================================
# DONE
# ============================================================
echo ""
echo "======================================"
echo " SETUP COMPLETE!"
echo "======================================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""
echo "To activate environment:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo "  # Or just: source ~/gaming-vfisr/activate.sh"
echo ""
echo "To copy videos from Windows:"
echo "  cp /mnt/c/Users/YOUR_NAME/Videos/gameplay.mp4 ~/gaming-vfisr/data/raw/"
echo ""
echo "IMPORTANT REMINDERS:"
echo "  - Work in WSL filesystem (~/), not /mnt/c/ for best performance"
echo "  - VFIMamba/SPAN weights need manual download"
echo ""
echo "Next steps:"
echo "  1. Copy your gaming footage to data/raw/"
echo "  2. Run preprocessing script"
echo "  3. Run benchmarks"
echo ""
echo "======================================"
