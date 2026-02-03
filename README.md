# Pocket TTS Arabic Adaptation (ClArTTS)

This repo contains two fine-tuning pipelines:

1. **Unsloth / Orpheus LoRA (recommended, matches Unsloth docs).**
2. Legacy **Pocket TTS** proxy flow-matching loop for experimentation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation (shared)

Downloads a small subset of MBZUAI/ClArTTS, normalizes Arabic text, romanizes it (Buckwalter), resamples audio, and writes `metadata.jsonl`.

```bash
python data_prep.py --streaming --max-samples 50
```

Outputs:
- `data/processed/metadata.jsonl`
- `data/processed/*.wav`
- `samples/reference.wav`

## Training (Unsloth / Orpheus)

Fine-tune the Unsloth Orpheus 3B model with a LoRA adapter. Defaults match the
[Unsloth TTS fine-tuning guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning).

```bash
python unsloth_train.py \
  --dataset MrDragonFox/Elise \
  --text-column transcription \
  --audio-column audio \
  --num-epochs 1 \
  --batch-size 8 \
  --grad-accum 2
```

Outputs:
- `checkpoints_unsloth/` with LoRA adapter + tokenizer.

## Training (Proxy Flow-Matching, Pocket TTS)

This script loads the Pocket TTS weights and fine-tunes the flow LM on Mimi latents using a proxy flow-matching objective. By default it trains only the text embedding; use `--train-scope text+flow` to fine-tune the full flow LM.

```bash
# Standard flow-matching
python train.py --metadata data/processed/metadata.jsonl --epochs 1 --max-steps 50 --train-scope text

# Flow-matching with lightweight LoRA adapters on the flow LM (closer to the Unsloth recipe)
python train.py --metadata data/processed/metadata.jsonl --use-lora --lora-rank 16 --lora-alpha 16 --lora-dropout 0.05
```

Checkpoints are saved to `checkpoints/`.

## Generate Arabic Sample

Generate a sample using the fine-tuned checkpoint (or base model if omitted). Use `--romanize` to romanize the Arabic text before inference.

```bash
python generate_sample.py --checkpoint checkpoints/flow_lm_final.pt --romanize
```

Outputs:
- `samples/generated_arabic.wav`

**Inference tip:** Training data is latinized with `simple_latin_transliterate` (chat-style 3/7). Default generation now auto-transliterates Arabic input the same way. Use `--latinize` to force it, or `--no-auto-transliterate` to pass raw Arabic. Use `--romanize` only if you specifically want Buckwalter.

Note: The repo includes placeholder WAVs; re-run `data_prep.py` and `generate_sample.py`
to regenerate real samples from the dataset and model.

## Files

- `exploration_report.md`: architecture notes and adaptation strategy.
- `data_prep.py`: dataset download + preprocessing.
- `train.py`: fine-tuning loop.
- `generate_sample.py`: inference script.
- `samples/`: output audio samples.
