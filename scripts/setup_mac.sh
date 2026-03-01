#!/usr/bin/env bash
# =============================================================================
# DefectBot — Mac Setup Script
# Installs and starts Ollama + WhisperKit for on-device AI inference.
#
# Usage:  bash scripts/setup_mac.sh
# Safe to run multiple times (idempotent).
#
# What this does:
#   1. Installs Ollama (local LLM server) via Homebrew
#   2. Installs WhisperKit CLI (local speech-to-text) via Homebrew
#   3. Downloads the Whisper small.en model (~250MB)
#   4. Starts Ollama server in background (port 11434)
#   5. Starts WhisperKit server in background (port 2022)
#   6. Prints next steps for loading the GGUF model
# =============================================================================
set -euo pipefail

# ── Colours for output ───────────────────────────────────────────────────────
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # no colour

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${ORANGE}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_section() { echo -e "\n${BLUE}══════════════════════════════════════════${NC}"; echo -e "${BLUE}  $*${NC}"; echo -e "${BLUE}══════════════════════════════════════════${NC}"; }

# ── Check: macOS ─────────────────────────────────────────────────────────────
if [[ "$(uname)" != "Darwin" ]]; then
  log_error "This script is for macOS only."
  exit 1
fi

log_section "DefectBot Mac Setup"
echo "  Target: Ollama (port 11434) + WhisperKit (port 2022)"
echo "  All services run locally — no internet connection required after setup."
echo ""

# ── Step 1: Homebrew check ───────────────────────────────────────────────────
log_section "Step 1/5 — Homebrew"
if ! command -v brew &>/dev/null; then
  log_error "Homebrew not found. Install from https://brew.sh then re-run this script."
  exit 1
fi
log_ok "Homebrew found: $(brew --version | head -1)"

# ── Step 2: Install Ollama ───────────────────────────────────────────────────
log_section "Step 2/5 — Ollama"

if command -v ollama &>/dev/null; then
  log_ok "Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
  log_info "Installing Ollama via Homebrew..."
  brew install ollama
  log_ok "Ollama installed"
fi

# ── Step 3: Install WhisperKit CLI ───────────────────────────────────────────
log_section "Step 3/5 — WhisperKit CLI"

if command -v whisperkit-cli &>/dev/null; then
  log_ok "whisperkit-cli already installed"
else
  log_info "Attempting: brew install whisperkit-cli"
  if brew install whisperkit-cli 2>/dev/null; then
    log_ok "whisperkit-cli installed via Homebrew"
  else
    log_warn "whisperkit-cli not in Homebrew — trying direct download..."

    # Direct download fallback (check WhisperKit GitHub releases)
    WHISPER_BIN="$HOME/.local/bin/whisperkit-cli"
    mkdir -p "$(dirname "$WHISPER_BIN")"

    # Determine architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
      WHISPER_URL="https://github.com/argmaxinc/WhisperKit/releases/latest/download/whisperkit-cli-macos-arm64.tar.gz"
    else
      WHISPER_URL="https://github.com/argmaxinc/WhisperKit/releases/latest/download/whisperkit-cli-macos-x86_64.tar.gz"
    fi

    log_info "Downloading from: $WHISPER_URL"
    if curl -fsSL "$WHISPER_URL" | tar -xz -C "$(dirname "$WHISPER_BIN")" 2>/dev/null; then
      chmod +x "$WHISPER_BIN"
      # Add to PATH for this session
      export PATH="$HOME/.local/bin:$PATH"
      log_ok "whisperkit-cli installed to $WHISPER_BIN"
    else
      log_warn "Could not install WhisperKit automatically."
      log_warn "Voice input will be unavailable. Text and image input still work."
      log_warn "Manual install: https://github.com/argmaxinc/WhisperKit/releases"
      SKIP_WHISPER=true
    fi
  fi
fi

SKIP_WHISPER=${SKIP_WHISPER:-false}

# ── Step 4: Whisper model — downloaded automatically by serve on first run ────
log_section "Step 4/5 — Whisper Model"

if [[ "$SKIP_WHISPER" == "false" ]] && command -v /opt/homebrew/bin/whisperkit-cli &>/dev/null; then
  log_ok "whisperkit-cli ready — model (small.en) will download automatically on first serve start"
else
  log_warn "whisperkit-cli not found at /opt/homebrew/bin/whisperkit-cli"
fi

# ── Step 5: Start services ───────────────────────────────────────────────────
log_section "Step 5/5 — Start Services"

# Start Ollama
if pgrep -x "ollama" &>/dev/null; then
  log_ok "Ollama already running"
else
  log_info "Starting Ollama server (port 11434, CORS=*)..."
  OLLAMA_ORIGINS="*" ollama serve >>/tmp/ollama-defectbot.log 2>&1 &
  OLLAMA_PID=$!
  log_ok "Ollama started (PID $OLLAMA_PID) — log: /tmp/ollama-defectbot.log"
  # Wait for Ollama to be ready
  log_info "Waiting for Ollama to become ready..."
  for i in {1..15}; do
    sleep 1
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
      log_ok "Ollama is ready at http://localhost:11434"
      break
    fi
    if [[ $i == 15 ]]; then
      log_warn "Ollama may still be starting — check /tmp/ollama-defectbot.log"
    fi
  done
fi

# Start WhisperKit server (on port 2023; CORS proxy on 2022 forwards to it)
if [[ "$SKIP_WHISPER" == "false" ]] && [[ -f "/opt/homebrew/bin/whisperkit-cli" ]]; then
  # Kill any stale process on port 2023
  if lsof -ti:2023 &>/dev/null; then
    log_ok "Port 2023 already in use (WhisperKit backend likely running)"
  else
    log_info "Starting WhisperKit backend on port 2023..."
    log_info "(First run will download small.en model ~250MB — this may take a minute)"
    /opt/homebrew/bin/whisperkit-cli serve --model small.en --port 2023 --host 127.0.0.1 >>/tmp/whisperkit-defectbot.log 2>&1 &
    WHISPER_PID=$!
    log_ok "WhisperKit started (PID $WHISPER_PID) — log: /tmp/whisperkit-defectbot.log"
  fi

  # CORS proxy: sits on port 2022 (what the app calls), forwards to WhisperKit on 2023
  # Needed because browsers block cross-origin responses without Access-Control-Allow-Origin.
  if lsof -ti:2022 &>/dev/null; then
    log_ok "Port 2022 already in use (CORS proxy likely running)"
  else
    log_info "Starting WhisperKit CORS proxy on port 2022..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python3 "$SCRIPT_DIR/whisper_cors_proxy.py" >>/tmp/whisperkit-defectbot.log 2>&1 &
    PROXY_PID=$!
    log_ok "CORS proxy started (PID $PROXY_PID)"
  fi
else
  log_warn "WhisperKit not started — voice input unavailable"
fi

# ── Next Steps ───────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Mac setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1️⃣  Copy your GGUF file from the training VM:"
echo "      scp user@vm-ip:~/defectbot/outputs/defectbot-gguf/*.gguf \\"
echo "          $(pwd)/outputs/defectbot-gguf/"
echo ""
echo "  2️⃣  Create a Modelfile in the project root:"
echo "      cat > Modelfile << 'EOF'"
echo "      FROM ./outputs/defectbot-gguf/defectbot-unsloth.Q4_K_M.gguf"
echo "      PARAMETER temperature 0.7"
echo "      PARAMETER num_ctx 4096"
echo "      SYSTEM \"You are DefectBot, an AI assistant for building and mining defect inspection.\""
echo "      EOF"
echo ""
echo "  3️⃣  Load the model into Ollama:"
echo "      ollama create defectbot -f Modelfile"
echo ""
echo "  4️⃣  Test the model:"
echo "      ollama run defectbot \"I see water dripping from the ceiling.\""
echo ""
echo "  5️⃣  Open the UI:"
echo "      open app/index.html"
echo ""
echo "  6️⃣  iPhone setup (optional):"
echo "      - Find Mac IP: ifconfig | grep 'inet ' | grep -v 127"
echo "      - In app/index.html, change localhost to your Mac's IP"
echo "        (OLLAMA_URL and WHISPER_URL at top of <script> section)"
echo ""
echo "  Logs:"
echo "      tail -f /tmp/ollama-defectbot.log"
echo "      tail -f /tmp/whisperkit-defectbot.log"
echo ""
