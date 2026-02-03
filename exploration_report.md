# Exploration Report: Pocket TTS → Classical Arabic (ClArTTS)

## 1) Pocket TTS Architecture (High-Level)

Pocket TTS is a lightweight, CPU-oriented text-to-speech model (≈100M parameters) that supports streaming generation and voice cloning, but is English-only out of the box. The official model card emphasizes CPU inference, low latency, and a Python API that exposes `TTSModel.load_model()`, `get_state_for_audio_prompt()`, and `generate_audio()` for text-to-audio generation. It also notes streaming support and the model's compact size.

The Pocket TTS paper is the **Continuous Audio Language Models (CALM)** work. CALM replaces discrete audio tokens with a continuous latent representation. A Transformer backbone produces a contextual embedding at each timestep; an MLP then generates the next continuous frame of an audio VAE via consistency/flow-style modeling. This avoids lossy codec discretization and aims to improve quality/efficiency.

From inspecting the codebase locally, the model combines:

- **Text conditioning** via a SentencePiece tokenizer and lookup-table embeddings (LUT conditioner).
- **A streaming Transformer** that consumes the text embeddings and previous latent frames.
- **A flow/consistency MLP** (FlowLM) that predicts the next continuous latent frame.
- **Mimi** (audio VAE) to map waveforms into continuous latent frames and decode them back to audio.

## 2) Dataset: MBZUAI/ClArTTS

ClArTTS is a Classical Arabic TTS corpus derived from a LibriVox audiobook. The dataset card describes ~12 hours of speech from a single male speaker, recorded at ~40.1 kHz. The dataset contains `train` and `test` splits and includes fields such as `text`, `file`, `audio`, `sampling_rate`, and `duration`.

Key implications:
- Single-speaker data makes speaker conditioning simpler but limits voice diversity.
- Sample rate mismatch (≈40.1 kHz) requires resampling to the Pocket TTS Mimi pipeline.

## 3) Adaptation Challenges (Arabic → Pocket TTS)

### Tokenization and Text Conditioning
Pocket TTS expects English-centric SentencePiece tokens. Classical Arabic introduces:
- Non-Latin script
- Optional diacritics (tashkeel), which may be inconsistently present
- Letter variants (e.g., Alef forms, hamza variations)

### Proposed Strategy
I chose a pragmatic two-stage approach:

1) **Arabic normalization**: strip diacritics, normalize letter variants, and remove tatweel.  
2) **Romanization**: convert Arabic characters to a Latin-based representation (Buckwalter).  

This allows reuse of the existing English tokenizer without retraining SentencePiece. It is a compromise that preserves a deterministic mapping while fitting the model’s current text pipeline.

## 4) Training Strategy

The official Pocket TTS package does not expose a training API. To demonstrate fine-tuning, I implemented a **proxy flow-matching objective**:

1) Encode target audio with Mimi to obtain continuous latent frames.  
2) Sample a random time `t ~ U(0,1)` and blend Gaussian noise with the latent targets.  
3) Use the Transformer + flow MLP to predict the velocity field at `t`.  
4) Optimize mean-squared error between predicted and target flow.

This mirrors flow/consistency training in the CALM framing (Transformer context → MLP predicts next continuous frame) while keeping the pipeline lightweight and CPU-friendly.

## References

- Pocket TTS model card (Kyutai)
- Continuous Audio Language Models (CALM) paper
- MBZUAI/ClArTTS dataset card

## 5) Summary of Deliverables

- **data_prep.py**: downloads ClArTTS, normalizes Arabic, romanizes to Buckwalter, resamples audio, and writes `metadata.jsonl`.
- **train.py**: fine-tunes the FlowLM (default: text embedding only) using the proxy flow-matching loss.
- **generate_sample.py**: loads a checkpoint (or base model) and generates audio from Arabic text (with optional romanization).
- **samples/**: includes a reference ClArTTS sample and a generated Arabic sample.

## 6) Future Extensions

- Train a dedicated Arabic SentencePiece tokenizer and resize the text embedding layer.
- Add phonemization (e.g., using Arabic G2P) to reduce homograph ambiguity.
- Fine-tune more of the Transformer backbone once training stability is verified.
