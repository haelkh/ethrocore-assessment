import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from scipy.signal import resample_poly
from tqdm import tqdm

from arabic_utils import buckwalter_transliterate, normalize_arabic, simple_latin_transliterate


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.shape[0] <= 4:
        return audio.mean(axis=0)
    return audio.mean(axis=-1)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def _extract_audio(sample) -> tuple[np.ndarray, int]:
    audio = sample["audio"]
    if isinstance(audio, dict) and "array" in audio:
        return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
    if hasattr(audio, "array"):
        return np.asarray(audio.array, dtype=np.float32), int(audio.sampling_rate)
    if isinstance(audio, list):
        sr = sample.get("sampling_rate")
        if sr is None:
            raise ValueError("Missing sampling_rate for list-based audio sample.")
        return np.asarray(audio, dtype=np.float32), int(sr)
    raise ValueError(f"Unsupported audio format in dataset sample: {type(audio)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MBZUAI/ClArTTS")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    metadata_path = output_dir / "metadata.jsonl"

    count = 0
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        iterator = dataset if args.streaming else tqdm(dataset, desc="processing")
        for sample in iterator:
            raw_text = sample.get("text", "").strip()
            if not raw_text:
                continue
            text_norm = normalize_arabic(raw_text)
            text_romanized = simple_latin_transliterate(text_norm)

            audio, orig_sr = _extract_audio(sample)
            audio = _to_mono(audio)
            audio = _resample(audio, orig_sr, args.target_sr)

            item_id = sample.get("id", f"sample_{count:06d}")
            audio_path = output_dir / f"{item_id}.wav"
            sf.write(audio_path, audio, args.target_sr)

            if count == 0:
                reference_path = samples_dir / "reference.wav"
                sf.write(reference_path, audio, args.target_sr)

            record = {
                "id": item_id,
                "text": raw_text,
                "text_normalized": text_norm,
                "text_romanized": text_romanized,
                "audio_path": str(audio_path),
                "sample_rate": args.target_sr,
            }
            meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            count += 1
            if count >= args.max_samples:
                break

    print(f"Wrote {count} samples to {output_dir}")


if __name__ == "__main__":
    main()
