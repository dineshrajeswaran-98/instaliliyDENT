# DefectBot — Claude Instructions

## What This Project Is
AI-powered building and mining defect inspection chatbot. Field technicians upload images
or speak defect descriptions; the bot asks follow-up questions, diagnoses root causes, and
generates professional manager summary reports.

## Deployment Architecture
- Mac: Ollama (local LLM) + WhisperKit (local STT), both on-device
- iPhone: connects to Mac's Ollama over LAN WiFi
- No cloud, no internet required

## Key Files
| File | Purpose |
|------|---------|
| `data/make_dataset.py` | Generates `data/train.jsonl` with 60+ synthetic training samples |
| `training/train.py` | Unsloth QLoRA fine-tuning → exports LoRA + GGUF Q4_K_M |
| `app/index.html` | Single-file production chat UI (vanilla JS, no deps) |
| `scripts/setup_mac.sh` | One-shot Mac setup: Ollama + WhisperKit |
| `scripts/setup_vm.sh` | One-shot VM setup: CUDA + Python ML deps |
| `outputs/` | Training outputs: LoRA weights + GGUF model files |

## Key Ports & Endpoints
- Ollama: `http://localhost:11434/api/chat` (model name: `defectbot`)
- WhisperKit: `http://localhost:2022/v1/audio/transcriptions` (OpenAI-compatible)

## LLM Configuration
- Base model: `google/gemma-3-4b-it`
- Fine-tuned model: `defectbot` (via Ollama)
- LoRA: rank=32, alpha=64, 4-bit QLoRA
- GGUF export: Q4_K_M quantization

## Training Data Format
Each JSONL line: `{"instruction": "...", "response": "..."}`
Response always ends with MANAGER SUMMARY section (Severity/Issue/Root Cause/Action/Timeline/Cost).

## Defect Categories
Building: AC condensation, pipeline leaks, structural cracks, spalling, efflorescence,
mould, paint deterioration, roof leakage, rising damp

Mining: conveyor belt tear/wear/deviation, idler roller faults, tunnel lining cracks,
tunnel seepage, hydraulic oil leaks, equipment oil puddles

## UI Demo Mode
If Ollama is not running, the UI automatically switches to demo mode with canned responses.
This allows the UI to be demoed without the model loaded.
