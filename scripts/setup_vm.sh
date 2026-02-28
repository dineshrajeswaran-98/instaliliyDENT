#!/usr/bin/env bash
# =============================================================================
# DefectBot — Linux/VM Training Environment Setup Script
# Sets up a CUDA-enabled Python environment for fine-tuning Gemma 3 4B.
#
# Tested on: Ubuntu 22.04 LTS with CUDA 12.1 (A100, H100, RTX 3090/4090)
# Minimum requirements: 16GB RAM, CUDA-enabled GPU with 16GB+ VRAM
# Recommended: A100 40GB or H100 for 5-epoch training in reasonable time
#
# Usage:  bash scripts/setup_vm.sh
# Safe to run multiple times (idempotent).
#
# What this does:
#   1. Updates apt and installs system dependencies
#   2. Creates a Python virtual environment: defectbot-env/
#   3. Installs PyTorch with CUDA 12.1
#   4. Installs Unsloth + all ML training dependencies
#   5. Creates project directory structure
# =============================================================================
set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${ORANGE}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_section() { echo -e "\n${BLUE}══════════════════════════════════════════${NC}"; echo -e "${BLUE}  $*${NC}"; echo -e "${BLUE}══════════════════════════════════════════${NC}"; }

log_section "DefectBot VM Setup"
echo "  Environment: Linux (Ubuntu 22.04+ recommended)"
echo "  CUDA version target: 12.1"
echo ""

# ── Step 1: System packages ───────────────────────────────────────────────────
log_section "Step 1/5 — System Dependencies"

log_info "Updating apt package list..."
sudo apt-get update -y -q

log_info "Installing system dependencies..."
sudo apt-get install -y -q \
  python3 \
  python3-pip \
  python3-venv \
  python3-dev \
  git \
  wget \
  curl \
  build-essential \
  libssl-dev \
  libffi-dev \
  screen \
  htop \
  nvtop 2>/dev/null || true   # nvtop may not be available on all distros

log_ok "System packages installed"

# ── Step 2: Check CUDA ───────────────────────────────────────────────────────
log_section "Step 2/5 — CUDA Check"

if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' 2>/dev/null || echo "unknown")
  log_ok "GPU detected: $GPU_INFO"
  log_ok "CUDA driver version: $CUDA_VER"

  # Warn if CUDA version doesn't match target
  if [[ "$CUDA_VER" != "12."* ]]; then
    log_warn "CUDA version $CUDA_VER detected. This script installs PyTorch for CUDA 12.1."
    log_warn "If your CUDA version is different, edit TORCH_INDEX_URL below."
  fi
else
  log_warn "nvidia-smi not found — no GPU detected or driver not installed."
  log_warn "Training will use CPU (VERY slow). For GPU training, use a CUDA-enabled VM."
fi

# ── Step 3: Python virtual environment ───────────────────────────────────────
log_section "Step 3/5 — Python Virtual Environment"

VENV_DIR="$(pwd)/defectbot-env"

if [[ -d "$VENV_DIR" ]]; then
  log_ok "Virtual environment already exists at $VENV_DIR"
else
  log_info "Creating virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
  log_ok "Virtual environment created"
fi

# Activate for the rest of this script
source "$VENV_DIR/bin/activate"
log_ok "Virtual environment activated: $(which python3)"

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools -q
log_ok "pip upgraded to $(pip --version)"

# ── Step 4: PyTorch with CUDA 12.1 ───────────────────────────────────────────
log_section "Step 4/5 — PyTorch Installation"

# Check if PyTorch already installed with CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log_ok "PyTorch with CUDA already installed: $(python3 -c 'import torch; print(torch.__version__)')"
else
  log_info "Installing PyTorch 2.x with CUDA 12.1..."
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

  pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url "$TORCH_INDEX_URL" \
    -q

  # Verify CUDA available after install
  if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    log_ok "PyTorch installed: $TORCH_VER with CUDA support"
  else
    log_warn "PyTorch installed but CUDA not available — training will use CPU"
  fi
fi

# ── Step 5: ML Training Libraries ────────────────────────────────────────────
log_section "Step 5/5 — ML Training Libraries"

log_info "Installing Unsloth and training dependencies..."
log_info "(This may take 5–15 minutes depending on your connection speed)"

# Install Unsloth — handles its own CUDA-compatible build
pip install "unsloth[colab-new]" -q 2>/dev/null || \
pip install "unsloth" -q

# Core training stack
pip install \
  trl \
  transformers \
  peft \
  accelerate \
  bitsandbytes \
  datasets \
  huggingface_hub \
  sentencepiece \
  protobuf \
  -q

log_ok "All ML libraries installed"

# Print installed versions
log_info "Installed versions:"
python3 -c "
import importlib
libs = ['torch', 'transformers', 'peft', 'trl', 'accelerate', 'bitsandbytes', 'datasets']
for lib in libs:
    try:
        m = importlib.import_module(lib)
        ver = getattr(m, '__version__', 'unknown')
        print(f'  {lib}: {ver}')
    except ImportError:
        print(f'  {lib}: NOT INSTALLED')
"

# ── Create project directory structure (idempotent) ──────────────────────────
log_section "Project Directory Structure"

mkdir -p defectbot/{data,training,app,scripts,outputs/defectbot-lora,outputs/defectbot-gguf}
log_ok "Project directories created (idempotent)"

# ── HuggingFace login reminder ────────────────────────────────────────────────
echo ""
echo -e "${ORANGE}NOTE: Gemma 3 requires HuggingFace access.${NC}"
echo "      If you haven't already, run: huggingface-cli login"
echo "      Then accept the Gemma license at: https://huggingface.co/google/gemma-3-4b-it"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ VM setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1️⃣  Activate the environment (in each new terminal session):"
echo "      source defectbot-env/bin/activate && cd defectbot"
echo ""
echo "  2️⃣  (First time only) Authenticate with HuggingFace:"
echo "      huggingface-cli login"
echo ""
echo "  3️⃣  Generate training data:"
echo "      python data/make_dataset.py"
echo ""
echo "  4️⃣  Start training inside screen (so it survives SSH disconnect):"
echo "      screen -S defectbot"
echo "      python training/train.py"
echo "      # Detach with Ctrl+A, D"
echo "      # Reattach with: screen -r defectbot"
echo ""
echo "  5️⃣  After training, copy GGUF to Mac:"
echo "      scp outputs/defectbot-gguf/*.gguf your-mac:~/Desktop/defectbot/outputs/defectbot-gguf/"
echo ""
