import argparse
import json
from pathlib import Path

import torch
import soundfile as sf
from tqdm import tqdm

from pocket_tts import TTSModel
from scipy.signal import resample_poly


def load_audio(path: Path, target_sr: int) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        gcd = torch.gcd(torch.tensor(sr), torch.tensor(target_sr)).item()
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype("float32")
    return torch.from_numpy(audio)[None, None, :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="data/processed/metadata.jsonl")
    p.add_argument("--output-dir", default="data/latents")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TTSModel.load_model()
    model.to(args.device)
    model.mimi.eval()

    records = []
    with open(args.metadata, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    if args.max_samples:
        records = records[: args.max_samples]

    meta_out = out_dir / "metadata_latents.jsonl"
    with meta_out.open("w", encoding="utf-8") as fout:
        for rec in tqdm(records, desc="encoding latents"):
            wav_path = Path(rec["audio_path"])
            audio = load_audio(wav_path, rec.get("sample_rate", 24000)).to(args.device)
            with torch.no_grad():
                latents = model.mimi.encode_to_latent(audio)
                latents = latents.transpose(-1, -2).to(torch.float32).cpu()
            latent_path = out_dir / f"{Path(rec['audio_path']).stem}_latents.pt"
            torch.save({"latents": latents, "length": latents.shape[1]}, latent_path)
            rec_out = rec.copy()
            rec_out["latent_path"] = str(latent_path)
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
    print(f"Saved latents and metadata to {out_dir}")


if __name__ == "__main__":
    main()
