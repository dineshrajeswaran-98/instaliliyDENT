#!/usr/bin/env python3
"""
DefectBot — Unsloth QLoRA Fine-tuning Script
Fine-tunes google/gemma-3-4b-it on defect inspection data.

Usage:
    python training/train.py              # full training run
    python training/train.py --dry-run   # validate setup without training

Outputs:
    outputs/defectbot-lora/   ← LoRA adapter weights
    outputs/defectbot-gguf/   ← GGUF Q4_K_M quantized model for Ollama
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="DefectBot fine-tuning script")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate data loading and model config without running full training",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=5,
    help="Number of training epochs (default: 5)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=2,
    help="Per-device training batch size (default: 2)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# GPU availability check
# ---------------------------------------------------------------------------
print("=" * 60)
print("DefectBot Training Script")
print("=" * 60)

try:
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        USE_GPU = True
    else:
        print("⚠️  WARNING: No CUDA GPU detected. Training will run on CPU.")
        print("   Expected training time on CPU: 24–72 hours (not recommended).")
        print("   Use a CUDA-enabled VM/cloud instance for practical training.")
        USE_GPU = False
except ImportError:
    print("❌ PyTorch not installed. Run: pip install torch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Project root is two levels up from this script (training/train.py -> root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train.jsonl")
LORA_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "defectbot-lora")
GGUF_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "defectbot-gguf")

os.makedirs(LORA_OUTPUT, exist_ok=True)
os.makedirs(GGUF_OUTPUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Validate training data
# ---------------------------------------------------------------------------
print(f"\n📂 Loading training data from: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"❌ Training data not found at {DATA_PATH}")
    print("   Run first: python data/make_dataset.py")
    sys.exit(1)

samples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            assert "instruction" in sample and "response" in sample, \
                f"Line {i+1} missing 'instruction' or 'response' key"
            samples.append(sample)
        except (json.JSONDecodeError, AssertionError) as e:
            print(f"❌ Error in data file at line {i+1}: {e}")
            sys.exit(1)

print(f"✅ Loaded {len(samples)} training samples")

if args.dry_run:
    print("\n✅ DRY RUN complete — data validation passed.")
    print(f"   Model: google/gemma-3-4b-it")
    print(f"   LoRA rank: 32, alpha: 64")
    print(f"   Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"   LoRA output: {LORA_OUTPUT}")
    print(f"   GGUF output: {GGUF_OUTPUT}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Import ML libraries (only after dry-run check to fail fast)
# ---------------------------------------------------------------------------
print("\n📦 Loading ML libraries...")

try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
except ImportError:
    print("❌ Unsloth not installed. Run: pip install unsloth")
    sys.exit(1)

try:
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Run: pip install trl transformers datasets")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-3-4b-it"
MAX_SEQ_LEN = 2048   # sufficient for instruction + response pairs
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0

# Target modules for Gemma architecture
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

print(f"\n🤖 Loading base model: {MODEL_NAME}")
print(f"   4-bit quantization: enabled")
print(f"   Max sequence length: {MAX_SEQ_LEN}")

# ---------------------------------------------------------------------------
# Load model and tokenizer via Unsloth
# ---------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,          # auto-detect: bf16 on Ampere+, fp16 on older GPUs
    load_in_4bit=True,   # 4-bit QLoRA reduces VRAM requirements
)

print("✅ Base model loaded")

# ---------------------------------------------------------------------------
# Apply LoRA adapters
# ---------------------------------------------------------------------------
print(f"\n🔧 Applying LoRA adapters (rank={LORA_RANK}, alpha={LORA_ALPHA})")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
    use_rslora=False,     # Rank-stabilised LoRA (optional)
    loftq_config=None,
)

# Print trainable parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ LoRA adapters applied")
print(f"   Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"   Total parameters:     {total_params:,}")

# ---------------------------------------------------------------------------
# Format training data as ChatML instruction-response pairs
# ---------------------------------------------------------------------------
print(f"\n📝 Formatting {len(samples)} samples as ChatML...")

# System prompt baked into every training sample
SYSTEM_PROMPT = """You are DefectBot, an AI assistant specialised in building and mining defect inspection.
When a technician reports a defect:
1. First ask 2-3 targeted follow-up questions to gather key details about the defect
2. After receiving answers, provide a structured DIAGNOSIS with root cause and contributing factors
3. Always end with a MANAGER SUMMARY section containing:
   - Severity: [CRITICAL/HIGH/MEDIUM/LOW]
   - Issue description
   - Root Cause
   - Immediate Action Required
   - Recommended Timeline
   - Estimated Cost Impact
Keep responses professional and concise. Use specific technical terminology appropriate for the defect type."""


def format_sample(sample):
    """Format a training sample as a ChatML conversation string."""
    instruction = sample["instruction"]
    response = sample["response"]

    # ChatML format that Gemma uses
    text = (
        f"<start_of_turn>user\n{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n{response}<end_of_turn>"
    )
    return {"text": text}


# Apply formatting to all samples
formatted_samples = [format_sample(s) for s in samples]
dataset = Dataset.from_list(formatted_samples)

print(f"✅ Dataset formatted — {len(dataset)} samples ready")
print(f"   Example (truncated):\n   {formatted_samples[0]['text'][:200]}...")

# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------
USE_BF16 = is_bfloat16_supported() and USE_GPU

training_args = TrainingArguments(
    output_dir=LORA_OUTPUT,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,          # effective batch = 8
    warmup_steps=10,
    learning_rate=2e-4,
    bf16=USE_BF16,
    fp16=not USE_BF16 and USE_GPU,
    logging_steps=10,                        # print loss every 10 steps
    optim="adamw_8bit" if USE_GPU else "adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=42,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",                        # disable wandb/tensorboard
    dataloader_pin_memory=USE_GPU,
)

print(f"\n🎯 Training configuration:")
print(f"   Epochs:              {args.epochs}")
print(f"   Batch size:          {args.batch_size} × 4 gradient accumulation = {args.batch_size * 4} effective")
print(f"   Learning rate:       2e-4 (cosine decay)")
print(f"   Precision:           {'bf16' if USE_BF16 else 'fp16' if USE_GPU else 'fp32 (CPU)'}")
print(f"   Steps per epoch:     ~{len(dataset) // (args.batch_size * 4)}")
print(f"   Total steps:         ~{len(dataset) * args.epochs // (args.batch_size * 4)}")

# ---------------------------------------------------------------------------
# SFTTrainer setup and training
# ---------------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=2,
    packing=False,                    # no packing — each sample is a full conversation
    args=training_args,
)

print(f"\n🚀 Starting training...")
print(f"   Loss will be printed every 10 steps")
print(f"   LoRA weights saved after each epoch to: {LORA_OUTPUT}")
print("-" * 60)

# Run training
trainer_stats = trainer.train()

print("-" * 60)
print(f"✅ Training complete!")
print(f"   Total training time: {trainer_stats.metrics.get('train_runtime', 0):.0f} seconds")
print(f"   Samples/second:      {trainer_stats.metrics.get('train_samples_per_second', 0):.2f}")
print(f"   Final loss:          {trainer_stats.metrics.get('train_loss', 0):.4f}")

# ---------------------------------------------------------------------------
# Save LoRA adapters
# ---------------------------------------------------------------------------
print(f"\n💾 Saving LoRA adapters to: {LORA_OUTPUT}")
model.save_pretrained(LORA_OUTPUT)
tokenizer.save_pretrained(LORA_OUTPUT)
print(f"✅ LoRA adapters saved")

# ---------------------------------------------------------------------------
# Export GGUF (Q4_K_M quantization for Ollama)
# ---------------------------------------------------------------------------
print(f"\n📦 Exporting GGUF Q4_K_M to: {GGUF_OUTPUT}")
print(f"   This merges LoRA with base model and quantizes — takes 10–20 minutes...")

try:
    model.save_pretrained_gguf(
        GGUF_OUTPUT,
        tokenizer,
        quantization_method="q4_k_m",    # best balance of size and quality
    )
    print(f"✅ GGUF export complete!")

    # Print the GGUF file path for user reference
    gguf_files = [f for f in os.listdir(GGUF_OUTPUT) if f.endswith(".gguf")]
    if gguf_files:
        gguf_path = os.path.join(GGUF_OUTPUT, gguf_files[0])
        gguf_size = os.path.getsize(gguf_path) / 1e9
        print(f"   GGUF file: {gguf_path}")
        print(f"   File size: {gguf_size:.2f} GB")
except Exception as e:
    print(f"⚠️  GGUF export failed: {e}")
    print(f"   LoRA weights are still saved and can be converted separately.")
    print(f"   Use: llama.cpp convert-hf-to-gguf.py {LORA_OUTPUT}")

# ---------------------------------------------------------------------------
# Print Ollama setup instructions
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("NEXT STEPS — Load model into Ollama:")
print("=" * 60)
print(f"""
1. Copy GGUF to Mac:
   scp {GGUF_OUTPUT}/*.gguf your-mac:/path/to/defectbot/outputs/defectbot-gguf/

2. Create Modelfile on Mac (create a file named 'Modelfile'):
   FROM ./outputs/defectbot-gguf/defectbot-unsloth.Q4_K_M.gguf
   PARAMETER temperature 0.7
   PARAMETER num_ctx 4096
   SYSTEM "You are DefectBot, an AI assistant for building and mining defect inspection."

3. Load into Ollama:
   ollama create defectbot -f Modelfile

4. Test:
   ollama run defectbot "I see water dripping from the ceiling."

5. Open the UI:
   open defectbot/app/index.html
""")
print("=" * 60)
