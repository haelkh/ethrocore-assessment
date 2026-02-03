# Implementation Plan: AI Audio Developer Assessment

## Goal

Adapt Kyutai's "Pocket TTS" (approx 100M params) to synthesize Classical Arabic using the ClArTTS dataset.
**Constraint**: Completion within 5-8 hours.
**Success Criteria**: Working training pipeline, decreasing loss, audible Arabic samples (quality secondary to pipeline functionality).

---

## Phase 1: Reconnaissance & Environment Setup (Hour 0-1)

_Objective: Establish a working baseline and understand the model internals._

1.  **Repository Analysis (`kyutai-labs/pocket-tts`)**:
    - Clone and install dependencies.
    - **Architecture Audit**: Determine if it's a pure LLM-based TTS (predicting audio tokens) or a Diffusion/Flow matching model.
    - **Tokenizer Check**: Verify the text tokenizer. Is it BPE? Is it English-only?
    - **Audio Codec**: Confirm usage of **Mimi** or another neural codec. Identify the sample rate (likely 24kHz) and codebook size.
    - **Inference Test**: Run the provided inference script on CPU to generate an English sample. Verify `environment.yml` or `requirements.txt` consistency.

2.  **Dataset Acquisition (`MBZUAI/ClArTTS`)**:
    - Inspect `MBZUAI/ClArTTS` on Hugging Face.
    - Check total duration and sample quality.
    - Identify text column and audio column.

---

## Phase 2: Data Preprocessing Strategy (Hour 1-2.5)

_Objective: Prepare data for high-efficiency training._

1.  **Text Normalization (Arabic to Model-Ready)**:
    - **Challenge**: The model is likely pre-trained on English/Latin tokens.
    - **Strategy**: **Romanization/Buckwalter Transliteration**.
      - Instead of extending the tokenizer (which destabilizes pre-trained weights) or retraining embeddings from scratch (too slow), we will map Arabic phonemes/graphemes to Latin characters (e.g., `sim` -> `s-i-m`, `kitab` -> `k-i-t-a-b`).
      - Library: `uroman` or `buckwalter` or `lang-transliteration`.
      - _Backup Plan_: If the model uses a phonemizer output (IPA), use `espeak-ng` with `ar` language code.

2.  **Audio Processing**:
    - **Resampling**: Convert all ClArTTS audio to **24kHz** (or model native SR).
    - **Silence Trimming**: Remove leading/trailing silence to reduce wasted compute.
    - **Tokenization (Crucial Step)**:
      - If the model is an AudioLM (consumes audio tokens), we must pre-encode the audio using the codec (e.g., Mimi) and save the discrete codes.
      - _Why_: Encoding on-the-fly during training is too slow.
    - **Manifest Creation**: Generate a JSONL/CSV file linking `normalized_text` -> `audio_path` (or `audio_tokens_path`).

3.  **Script**: `data_prep.py`

---

## Phase 3: Model Adaptation & Training Pipeline (Hour 2.5-6)

_Objective: Fine-tune the model to align Arabic text features with Arabic audio tokens._

1.  **Adaptation Architecture**:
    - **Text Embedding**: If using Romanization, freeze the majority of the model and only fine-tune the cross-attention layers (if applicable) or the last few transformer blocks.
    - **Full Fine-Tuning (LoRA)**: If the model supports LoRA (Low-Rank Adaptation), use it. It's faster and requires less memory. If not, use standard fine-tuning but freeze the bottom encoder layers.

2.  **Training Script (`train.py`)**:
    - **Framework**: PyTorch + Hugging Face `Accelerate` (for mixed precision `fp16`/`bf16` scaling).
    - **Dataloader**: Custom dataset class to load (Text, Audio/Codes) pairs.
    - **Loss Function**: Identify the native loss (Cross Entropy for AudioLMs, MSE/L1 for regression-based).
    - **Optimization**: AdamW optimizer, linear warmup scheduler.
    - **Checkpointing**: Save every epoch.

3.  **Execution**:
    - Run for ~10-20 epochs on a subset first (sanity check).
    - Then run on full dataset (or as much as fits in time).
    - **Monitor**: WandB or simple CLI logging of Loss vs Steps.

---

## Phase 4: Inference & Delivery (Hour 6-8)

_Objective: Generate evidence of learning._

1.  **Inference Script (`generate_sample.py`)**:
    - Load the fine-tuned weights.
    - Accept Arabic text input.
    - Apply the _same_ normalization logic (Romanization) as training.
    - Generate audio tokens -> Decode via Codex (Mimi) -> Save `.wav`.

2.  **Report Generation (`exploration_report.md`)**:
    - Document the "Why Romanization?" decision clearly.
    - Explain the Codec mechanism.
    - Show Loss curve snippet.
    - Analyze quality (Robot voice vs intelligible).

## Technical "Ace" Factors (The Differentiators)

- **Audio-Code Caching**: Pre-calculating audio codes is a pro move that speeds up training 10x.
- **Buckwalter/Romanization**: Recognizing that English models can't "read" Arabic script without help is the key insight.
- **Mixed Precision**: Using `Accelerate` shows production-readiness.
- **Evaluation**: Even a subjective listening test recorded in the report adds value.

## File Structure Plan

```
.
├── data_prep.py            # Downloads ClArTTS, normalizes text, encodes audio
├── train.py                # Main training loop with Accelerate
├── generate_sample.py      # Inference script
├── exploration_report.md   # Analysis documentation
├── requirements.txt        # Reproducibility
└── samples/                # Evidence
    ├── reference.wav
    └── generated_arabic.wav
```
