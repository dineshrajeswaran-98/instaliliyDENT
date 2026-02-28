# DefectBot — AI-Powered Defect Inspection Chatbot

An on-device AI chatbot for building and mining defect inspection. Field technicians describe defects via text, image, or voice — DefectBot diagnoses root causes and generates professional manager reports.

**Everything runs locally. No cloud. No internet required after setup.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING (VM)                            │
│                                                                 │
│   data/make_dataset.py  ─→  data/train.jsonl (60+ samples)    │
│         ↓                                                       │
│   training/train.py     ─→  Unsloth QLoRA fine-tune            │
│         ↓                       (google/gemma-3-4b-it)         │
│   outputs/defectbot-lora/   (LoRA weights)                     │
│   outputs/defectbot-gguf/   (GGUF Q4_K_M)  ←─ copy to Mac    │
└─────────────────────────────────────────────────────────────────┘
                              │ SCP
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        MAC (On-Device)                          │
│                                                                 │
│   Ollama (:11434)   ←─  defectbot GGUF model                  │
│       ↑ /api/chat                                               │
│   app/index.html  ←── Safari / Chrome (open directly)          │
│       ↑ /v1/audio/transcriptions                                │
│   WhisperKit (:2022) ← small.en model                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │ WiFi (LAN)
                              ↓
┌──────────────────────────┐
│   iPhone (Optional)      │
│   Safari → Mac IP:port   │
│   Same index.html        │
└──────────────────────────┘
```

---

## Prerequisites

| Component | Mac | VM |
|-----------|-----|-----|
| OS | macOS 13+ (Apple Silicon recommended) | Ubuntu 22.04 LTS |
| RAM | 8GB minimum, 16GB recommended | 16GB minimum |
| GPU | Not required (Ollama uses Metal) | CUDA 12.x GPU, 16GB+ VRAM |
| Storage | 10GB free (for models) | 50GB free (model + dataset) |
| Software | Homebrew | Python 3.10+ |

**Recommended GPU for training:** A100 40GB (trains in ~20 min), H100 (faster), RTX 4090 (viable), RTX 3090 24GB (viable with batch=1).

---

## Complete Setup Guide

### Step 1 — VM Setup (Training Environment)

SSH into your GPU VM:

```bash
ssh user@your-vm-ip
```

Recommended: Use `screen` so training survives SSH disconnection:

```bash
# Install screen if needed
sudo apt-get install -y screen

# Start a named session
screen -S defectbot

# To detach (keep running): Ctrl+A, then D
# To reattach later:
screen -r defectbot
```

Clone or copy the project onto the VM, then run setup:

```bash
cd ~
git clone your-repo OR scp -r local/defectbot user@vm-ip:~/defectbot

cd defectbot
bash scripts/setup_vm.sh
```

This installs all dependencies. Takes 5–15 minutes.

---

### Step 2 — HuggingFace Authentication (First Time Only)

Gemma 3 is a gated model — you need to accept the license first.

```bash
# 1. Go to: https://huggingface.co/google/gemma-3-4b-it
#    Click "Agree and Access Repository"

# 2. Get your HF token: https://huggingface.co/settings/tokens

# 3. Login on the VM:
source defectbot-env/bin/activate
huggingface-cli login
# Paste your token when prompted
```

---

### Step 3 — Generate Training Data

```bash
source defectbot-env/bin/activate
cd defectbot

python data/make_dataset.py
```

Expected output:
```
✅ Dataset written to: data/train.jsonl
   Total samples: 64
   Building samples: 18
   Mining samples:   28
   Extra samples:    4
```

---

### Step 4 — Train the Model

```bash
# Inside screen session (so it survives disconnect)
screen -S defectbot

source defectbot-env/bin/activate
cd defectbot

python training/train.py
```

Monitor training progress — loss is printed every 10 steps:
```
{'loss': 1.42, 'learning_rate': 0.0002, 'epoch': 0.3}
{'loss': 1.18, 'learning_rate': 0.00019, 'epoch': 0.6}
...
{'loss': 0.31, 'learning_rate': 0.0, 'epoch': 5.0}
✅ Training complete!
```

**Training time estimates:**
| GPU | ~Time |
|-----|-------|
| A100 40GB | 15–25 minutes |
| RTX 4090 24GB | 25–45 minutes |
| RTX 3090 24GB | 45–90 minutes |
| CPU only | 24–72 hours (not recommended) |

To validate setup without full training:
```bash
python training/train.py --dry-run
```

---

### Step 5 — Copy GGUF to Mac

After training completes, copy the GGUF file to your Mac:

```bash
# On Mac (in Terminal):
scp user@vm-ip:~/defectbot/outputs/defectbot-gguf/*.gguf \
    ~/Desktop/defectbot/outputs/defectbot-gguf/
```

Verify the file is there:
```bash
ls -lh ~/Desktop/defectbot/outputs/defectbot-gguf/
# Should see: defectbot-unsloth.Q4_K_M.gguf (~2.5GB)
```

---

### Step 6 — Mac Setup

```bash
cd ~/Desktop/defectbot
bash scripts/setup_mac.sh
```

This installs Ollama and WhisperKit, downloads the Whisper model, and starts both servers.

---

### Step 7 — Load Model into Ollama

Create a `Modelfile` in the project root:

```bash
cat > Modelfile << 'EOF'
FROM ./outputs/defectbot-gguf/defectbot-unsloth.Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
PARAMETER num_predict 1024
SYSTEM "You are DefectBot, an expert AI assistant for building and mining defect inspection. Ask targeted follow-up questions first, then diagnose root cause, then produce a MANAGER SUMMARY with Severity, Issue, Root Cause, Immediate Action, Timeline, and Cost."
EOF
```

Load the model:
```bash
ollama create defectbot -f Modelfile
```

Test it:
```bash
ollama run defectbot "I see water dripping from the ceiling near the AC unit."
```

---

### Step 8 — Launch the UI

```bash
open app/index.html
```

Or double-click `app/index.html` in Finder. No server needed — it runs entirely in the browser.

The UI will:
- Automatically detect if Ollama is running
- Fall back to **demo mode** (canned responses) if Ollama is not available
- Show **On-Device** badge when connected to Ollama

---

### Step 9 — iPhone Setup (Optional)

To use DefectBot on iPhone over WiFi:

1. Find your Mac's local IP address:
   ```bash
   ifconfig | grep 'inet ' | grep -v 127
   # e.g. inet 192.168.1.42
   ```

2. Allow Ollama to accept LAN connections:
   ```bash
   # Stop current Ollama, restart with network binding
   pkill ollama
   OLLAMA_HOST=0.0.0.0 ollama serve &
   ```

3. Edit `app/index.html` — change these two lines:
   ```javascript
   const OLLAMA_URL   = 'http://192.168.1.42:11434/api/chat';   // ← Mac's IP
   const WHISPER_URL  = 'http://192.168.1.42:2022/v1/audio/transcriptions';  // ← Mac's IP
   ```

4. Copy the edited `index.html` to iPhone (AirDrop, or serve via:)
   ```bash
   cd app && python3 -m http.server 8080
   # Open Safari on iPhone: http://192.168.1.42:8080
   ```

---

## Using the UI

| Feature | How to use |
|---------|-----------|
| **Text input** | Type in the input box, press Enter to send |
| **Newline** | Shift+Enter |
| **Quick chips** | Click a chip to fill a pre-written defect description |
| **Image upload** | Click 📷, select an image — thumbnail appears in your message |
| **Voice input** | Hold 🎙️ to record, release to transcribe (auto-sends after 1.5s) |
| **Copy message** | Hover over any bubble → "Copy" button appears |
| **Export report** | After a diagnosis, click "⬇ Export Report" in the Manager Summary card |
| **Demo mode** | Works without Ollama — shows example responses |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| UI shows "DEMO MODE" | Ollama not running or not reachable | Run `ollama serve` then refresh page |
| `ollama run defectbot` fails | Model not loaded | Run `ollama create defectbot -f Modelfile` |
| Voice button not working | Mic permission denied or HTTPS required | Allow microphone in browser settings; try from localhost |
| WhisperKit not found via brew | Not yet in Homebrew | Setup script handles manual install automatically |
| Training OOM (out of memory) | GPU VRAM too small | Reduce `--batch-size 1` in train.py args |
| `huggingface_hub` auth error | Not logged in / no license | Run `huggingface-cli login`; accept Gemma license on HF |
| GGUF export fails | Unsloth version issue | Use LoRA weights directly; convert with llama.cpp separately |
| iPhone can't reach Mac | Firewall or wrong IP | Check Mac firewall, verify IP with `ifconfig`, use `OLLAMA_HOST=0.0.0.0` |

---

## File Reference

```
defectbot/
├── CLAUDE.md                  ← AI assistant instructions
├── Modelfile                  ← Ollama model definition (create after training)
├── data/
│   ├── make_dataset.py        ← Generates training data
│   └── train.jsonl            ← Generated: 60+ training samples (JSONL)
├── training/
│   └── train.py               ← Unsloth QLoRA fine-tuning script
├── app/
│   └── index.html             ← Complete chat UI (single file, no deps)
├── scripts/
│   ├── setup_mac.sh           ← Mac: installs Ollama + WhisperKit
│   └── setup_vm.sh            ← VM: installs Python ML environment
├── outputs/
│   ├── defectbot-lora/        ← Training output: LoRA adapter weights
│   └── defectbot-gguf/        ← Training output: GGUF model file
└── README.md                  ← This file
```

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Base model | google/gemma-3-4b-it |
| Fine-tuning method | QLoRA (4-bit) |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Epochs | 5 |
| Learning rate | 2e-4 (cosine decay) |
| GGUF quantization | Q4_K_M |
| Model size (GGUF) | ~2.5GB |
| VRAM for inference | ~3GB (via Ollama with Metal/CPU) |

---

*DefectBot — Built for field technicians. Runs on your device.*
