# Pocket TTS Arabic Adaptation (ClArTTS)

This repository contains a complete pipeline for adapting Kyutai's **Pocket TTS** model to synthesize **Classical Arabic**. It implements a custom **fine-tuning methodology** using conditional flow matching and **Romanization** to bridge the language gap.

## 1. Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Methodology

The adaptation relies on two key strategies:

1.  **Romanization (Arabizi)**: Mapping Arabic phonemes to Latin characters to leverage the model's pre-trained tokenizer and latent space.
2.  **Flow Matching Fine-Tuning**: Reverse-engineering the training loop to optimize the Flow Matching MLP (FlowLM) on Arabic audio latents.

Full architectural details are available in [exploration_report.md](exploration_report.md).

### 2.1 What are we training?

This is **not** teaching the model to read Arabic script. It is **fine-tuning the Flow Matching Network (FlowLM)** to learn an **Arabic accent** for Romanized text.

- **The Input**: Romanized Arabic (`marhaban`).
- **The Adjustment**: Weights are updated to map these English-compatible tokens to Arabic prosody and phonetics (throat sounds, timing) instead of American pronunciation.
- **The Mechanism**: We modify the Flow weights that predict the _trajectory_ of audio latents in the Mimi VAE space.

### 2.2 Inference Pipeline

1.  **Input**: Arabic Text (`مرحبا`)
2.  **Romanization**: Converts to `marhaban` (Arabizi).
3.  **Conditioning**: Model sees Latin characters and retrieves standard English embeddings.
4.  **Flow Prediction**: The fine-tuned FlowLM predicts the continuous path from noise to speech latents.
5.  **Decoding**: Mimi VAE converts these latents into the final Arabic audio waveform.

## 3. Usage

### 3.1 Data Preparation

Download and normalize the ClArTTS dataset. We explicitly **keep diacritics** to ensure high-fidelity pronunciation and prosody.

```bash
python data_prep.py --output-dir data/processed_arabizi --keep-diacritics
```

### 3.2 Optimization: Pre-computing Latents (Optional)

To speed up training by avoiding repetitive encoding, pre-compute the Mimi latents:

```bash
python extract_latents.py --metadata data/processed_arabizi/metadata.jsonl --output-dir data/latents
```

### 3.3 Training

Fine-tune the model using **LoRA (Low-Rank Adaptation)** and **Gradient Accumulation** for stability.
This configuration saves checkpoints and evaluates every 500 steps.

**Hyperparameters:**

- Batch Size: 2 (effective 8 with accumulation)
- LoRA Rank: 8
- Training Scope: Text Embeddings + Flow Network

```bash
# Standard training (on-the-fly encoding)
python train.py \
  --metadata data/processed_arabizi/metadata.jsonl \
  --checkpoint-dir checkpoints_diac_arabizi \
  --train-scope text+flow \
  --batch-size 2 \
  --grad-accum 4 \
  --save-every 500 \
  --eval-every 500 \
  --use-lora \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05

# OR if using pre-computed latents (faster):
python train.py \
  --metadata data/latents/metadata_latents.jsonl \
  --checkpoint-dir checkpoints_diac_latents \
  --use-precomputed-latents \
  --train-scope text+flow \
  --batch-size 8 \
  --grad-accum 1 \
  --use-lora \
  --lora-rank 8
```

### 3.4 Generation

Generate samples using a trained checkpoint. The `--latinize` flag ensures the input text is romanized using the same strategy (Arabizi) as the training data.

```bash
# Example: Using checkpoint at step 3500
python generate_sample.py \
  --checkpoint checkpoints_diac_arabizi/flow_lm_step_3500.pt \
  --text "مَرْحَبًا بكم" \
  --latinize \
  --no-arabizi \
  --append-translit
```

**Output:** `samples/generated_arabic_marhaban_bikum.wav`

## 4. Deliverables

- `exploration_report.md`: Comprehensive technical report and adaptation strategy.
- `data_prep.py`: Data download, normalization, and Romanization pipeline.
- `train.py`: Custom training loop implementing Conditional Flow Matching loss.
- `generate_sample.py`: Inference script with auto-transliteration.
- `samples/`: Directory containing reference and generated audio.
