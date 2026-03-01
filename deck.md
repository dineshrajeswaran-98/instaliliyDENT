# DefectBot
## AI-Powered Defect Inspection — On Device, On Site, Offline

> Field technicians get instant diagnoses and professional reports from a photo, a voice note, or a few words — with no internet, no cloud, no delays.

---

# The Problem

## Defect inspection today is broken

- Findings live in notebooks, voice memos, and photo rolls
- Root cause diagnosis requires a senior engineer — who isn't on site
- Reports are written hours or days later, from memory
- Critical defects go unescalated because the technician didn't know the severity

### The cost

| Gap | Impact |
|-----|--------|
| Delayed diagnosis | Defects worsen between inspection and expert review |
| Inconsistent reporting | Managers can't compare severity across sites |
| No internet underground / remote sites | Cloud AI tools are unavailable exactly when needed |
| Knowledge locked in senior engineers | Junior technicians under-equipped in the field |

---

# The Solution

## DefectBot — Your AI Inspection Expert in Your Pocket

A purpose-built AI chatbot that fits the way field technicians actually work.

- **Upload a photo** → instant visual analysis of what's wrong
- **Speak your observation** → voice-to-text, no typing required
- **Answer 2–3 quick questions** → tap clickable options, no typing
- **Get a diagnosis + manager report** → ready to share in seconds

### Built for two of the toughest environments

🏗️ **Building Inspection** — cracks, water ingress, spalling, mould, rising damp, roof failures

⛏️ **Mining Operations** — conveyor belts, idler rollers, tunnel linings, hydraulic leaks, heavy equipment

---

# How It Works

## The conversation flow

```
Technician uploads photo of cracked concrete column
       ↓
DefectBot: "I can see exposed rebar with rust layering across a 40cm spall.
            Is the rebar still solid or does it feel crumbly?
            Are there other columns nearby with similar map cracking?"

       [Solid]    [Crumbly / Pitting]          ← tap to answer

Technician: taps "Crumbly / Pitting"
       ↓
DefectBot: Diagnosis + Root Cause + ...

╔══════════════════════════════════╗
║  📋 Manager Summary              ║
║  Severity: HIGH                  ║
║  Issue: Active rebar corrosion…  ║
║  Immediate Action: …             ║
║  Timeline: 48 hours              ║
║  Cost Impact: $3,000–$8,000      ║
╚══════════════════════════════════╝
          [ ⬇ Export Report ]
```

---

# The Technology

## Fully on-device AI stack — no internet required

| Layer | Technology | Role |
|-------|-----------|------|
| Language Model | Gemma 3 4B (Google) | Diagnosis & report generation |
| Fine-Tuning | Unsloth QLoRA | Domain adaptation to defect inspection |
| Inference | Ollama | Local LLM server on Mac |
| Vision | Gemma 3 multimodal | Reads and interprets uploaded photos |
| Speech-to-Text | WhisperKit | On-device voice transcription |
| UI | Single-file HTML | Opens in any browser, zero install |

### Training pipeline

```
68 domain-specific examples
       ↓
QLoRA fine-tune on cloud GPU (~20 min on A100)
       ↓
GGUF Q4_K_M export (~3.3 GB)
       ↓
Loaded into Ollama on Mac — runs at full speed on Apple Silicon
```

---

# Architecture

## Mac as the brain, iPhone as the interface

```
┌─────────────────────────────────────┐
│           Mac (On-Device)           │
│                                     │
│  Ollama :11434  ←  defectbot GGUF  │
│  WhisperKit :2022  ← small.en STT  │
│  app/index.html  (served locally)   │
└──────────────┬──────────────────────┘
               │  WiFi (LAN)
               ↓
┌──────────────────────┐
│  iPhone — Safari     │
│  Same UI, same model │
│  No app install      │
└──────────────────────┘
```

### Why this architecture wins

- **No cloud costs** — zero API fees, runs forever
- **No data leaves the site** — inspections stay private
- **Works underground** — no signal needed
- **iPhone-ready** — one URL over LAN WiFi, nothing to install
- **Demo mode** — UI works without the model loaded, for demos

---

# The Model

## Fine-tuned for defect inspection — not a generic chatbot

**Base:** Google Gemma 3 4B (multimodal — sees images natively)

**Fine-tuned on 68 expert-crafted examples across:**

| Building | Mining |
|----------|--------|
| AC condensation & pipe leaks | Conveyor belt tears, wear & deviation |
| Structural cracks & foundation movement | Idler roller bearing failures |
| Concrete spalling & rebar corrosion | Tunnel lining cracks & seepage |
| Efflorescence & rising damp | Hydraulic hose & pump leaks |
| Roof membrane & waterproofing failures | Heavy equipment frame fatigue |
| Mould, paint failures, sealant breakdown | SAG mill vibration, pump station flooding |

**Every response always ends with a structured MANAGER SUMMARY** including severity rating, root cause, immediate action, timeline, and cost estimate.

---

# Key Features

## Built for field technicians

### Input
- 📷 **Photo upload** — drag, tap, or capture; vision model reads the image
- 🎙️ **Hold-to-record voice** — WhisperKit transcribes on-device, auto-sends after 1.5s
- ⌨️ **Text** — describe what you see in plain language
- **Quick chips** — tap a defect type to pre-fill the description

### Interaction
- **Clickable answer chips** — tap Q1: Wet / Dry instead of typing
- **Streaming responses** — answers appear token by token, no waiting

### Output
- **Diagnosis** — root cause and contributing factors
- **Manager Summary card** — structured severity + action + cost
- **Export Report** — downloads a timestamped `.txt` report
- **Copy button** — on every response bubble

---

# Demo

## What the technician sees

> *Technician holds up phone, taps 📷, takes a photo of a seized conveyor idler with a visible flat spot and blue heat tint*

**DefectBot responds in ~3 seconds:**

> I can see a seized return strand idler with an 8cm flat spot and blue-brown heat tint on the steel — there's a faint smoke trail visible at the belt contact point.
>
> 1. Is the conveyor currently running, or has it been stopped?
> 2. Is this roller in a section carrying coal or another combustible material?
> 3. Can you see or smell burning rubber on the belt underside?

**Chips appear:**

`Q1:` **[Still running]** **[Already stopped]**
`Q2:` **[Coal / combustible]** **[Non-combustible]**

> *Technician taps — done in 10 seconds*

**Result:** CRITICAL severity report, ready to share with the site manager.

---

# Why It Matters

## The numbers behind the problem

| Metric | Status Quo | With DefectBot |
|--------|-----------|----------------|
| Time from observation to written report | Hours to days | < 2 minutes |
| Severity correctly identified by junior tech | Inconsistent | Guided by AI |
| Reports requiring senior engineer review to draft | ~100% | Minimal |
| Works without internet | ❌ | ✅ |
| Data leaves the site | ✅ (cloud tools) | ❌ (on-device) |

---

# Roadmap

## Where this goes next

- **Retrain on larger dataset** — scale to 500+ examples with site-specific data
- **iOS native app** — SwiftUI wrapper for WhisperKit + camera integration
- **Photo annotation** — draw on the image to mark the defect location
- **Multi-defect session** — track multiple defects per inspection walkthrough
- **Export to PDF** — formatted report with embedded photo and site details
- **Sync when back on network** — offline-first with periodic cloud sync

---

# Built With

| | |
|-|-|
| **Model** | Google Gemma 3 4B (multimodal, vision-capable) |
| **Fine-tuning** | Unsloth QLoRA — 4-bit, rank 32, ~20 min on A100 |
| **Quantization** | GGUF Q4_K_M (~3.3 GB, runs on 8GB RAM) |
| **Inference** | Ollama (Mac, Apple Silicon Metal acceleration) |
| **STT** | WhisperKit small.en — on-device, < 1s latency |
| **Frontend** | Vanilla JS, single HTML file, zero dependencies |
| **Training data** | 68 hand-crafted examples across 16 defect categories |

> **Everything fits in a single HTML file and a 3.3 GB model.
> Open the file in a browser. That's the entire deployment.**
