# HAKI — API Keys & Environment Setup (2026 Architecture)

**IMPORTANT:** Never commit API keys to git. Add these to `~/.zshrc` only.

## Required keys

```bash
# ── LLM ──────────────────────────────────────────────────────────────────────
# Groq  — primary conversational LLM (Llama-3.3-70B, ~800 tok/s via LPU)
# https://console.groq.com/keys
export HAKI_GROQ_API_KEY="gsk_..."

# Cerebras — automatic rate-limit fallback for Groq (Llama-4-Scout, 2000+ TPS)
# https://cloud.cerebras.ai/
export HAKI_CEREBRAS_API_KEY="csk-..."

# Gemini — large-context tasks: RAG, PDF humanization, Obsidian vault (1M ctx)
# https://aistudio.google.com/app/apikey  (free tier sufficient)
export HAKI_GEMINI_API_KEY="AIza..."

# ── STT ──────────────────────────────────────────────────────────────────────
# Deepgram Flux — primary streaming STT with end-of-turn detection
# Free tier: $200 credits (~430h streaming)
# https://console.deepgram.com
export HAKI_DEEPGRAM_API_KEY="..."

# ── TTS ──────────────────────────────────────────────────────────────────────
# Cartesia Sonic 3.5 — cloud TTS fallback via Bengaluru Blue Machines (<15ms RTT)
# https://play.cartesia.ai/
export HAKI_CARTESIA_API_KEY="sk_car_..."

# ── Obsidian Vault ────────────────────────────────────────────────────────────
export HAKI_OBSIDIAN_VAULT="$HOME/Obsidian/HAKI_Brain"
```

## Local models (no key needed — install once)

| Component | Model | RAM | Install |
|-----------|-------|-----|---------|
| LLM primary | xLAM-2-3b-fc-r-4bit | ~1.85 GB | `mlx_lm.convert` or HF auto-download via mlx-lm |
| LLM standby | Bonsai-8B-mlx-1bit | ~1.28 GB | HF auto-download via mlx-lm |
| STT ANE | WhisperKit Tiny CoreML | ~40 MB | `pip install whisperkittools` |
| STT fallback | mlx-whisper tiny | ~40 MB | `pip install mlx-whisper` |
| TTS ANE | Kokoro-82M CoreML | ~170 MB | `pip install kokoro-onnx` |
| Embeddings | Granite ModernBERT 97M | ~97 MB | auto-downloaded by sentence-transformers |

System deps: `brew install espeak-ng`

## Total active RAM budget (worst case)

| Component | RAM |
|-----------|-----|
| xLAM-2-3b (local LLM) | 1.85 GB |
| Kokoro-82M CoreML (TTS) | ~0.2 GB |
| Granite embeddings (97M) | ~0.1 GB |
| WhisperKit Tiny (STT) | ~0.05 GB |
| Python runtime + ChromaDB | ~0.5 GB |
| **Total** | **~2.7 GB** |

Leaves ~5.3 GB for macOS + other apps — well within the 8 GB M2 ceiling.

## Quick start (~/.zshrc)

```bash
export HAKI_GROQ_API_KEY=""         # FILL IN
export HAKI_CEREBRAS_API_KEY=""     # FILL IN
export HAKI_GEMINI_API_KEY=""       # FILL IN
export HAKI_DEEPGRAM_API_KEY=""     # FILL IN
export HAKI_CARTESIA_API_KEY=""     # FILL IN
export HAKI_OBSIDIAN_VAULT="$HOME/Obsidian/HAKI_Brain"
```

After: `source ~/.zshrc`

Then install Python packages:
```bash
pip install -r Core/requirements.txt
```
