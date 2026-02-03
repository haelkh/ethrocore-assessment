import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm

from arabic_utils import buckwalter_transliterate, normalize_arabic, simple_latin_transliterate
try:
    from pocket_tts import TTSModel
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _Path

    local_repo = _Path(__file__).parent / "_pocket-tts"
    if local_repo.exists():
        sys.path.insert(0, str(local_repo))
    from pocket_tts import TTSModel
from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.modules.stateful_module import init_states
from scipy.signal import resample_poly


class TTSDataset(Dataset):
    def __init__(self, metadata_path: Path, use_latents: bool = False):
        self.records = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                self.records.append(json.loads(line))
        self.use_latents = use_latents

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx]


class LoRALinear(nn.Module):
    """Minimal, HF-independent LoRA wrapper for nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Freeze base weights/bias
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        # LoRA factors
        self.lora_A = nn.Parameter(torch.zeros((rank, base.in_features)))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        # Expose for modules that expect .weight (read-only)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, alpha: int, dropout: float):
        return cls(linear, rank=rank, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t()
        lora_out = lora_out @ self.lora_B.t()
        return base_out + lora_out * self.scaling


def _replace_linear_with_lora(module: nn.Module, rank: int, alpha: int, dropout: float) -> None:
    """Recursively replace Linear layers with LoRALinear in-place."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lora = LoRALinear.from_linear(child, rank, alpha, dropout)
            lora.to(child.weight.device)
            setattr(module, name, lora)
        else:
            _replace_linear_with_lora(child, rank, alpha, dropout)


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    val_size = max(1, int(n * val_ratio))
    return indices[val_size:], indices[:val_size]


def _load_audio(path: Path, target_sr: int, max_seconds: float | None) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
    if max_seconds is not None:
        max_len = int(target_sr * max_seconds)
        if audio.shape[0] > max_len:
            start = random.randint(0, audio.shape[0] - max_len)
            audio = audio[start : start + max_len]
    return torch.from_numpy(audio)


def _set_train_scope(flow_lm, scope: str) -> None:
    for param in flow_lm.parameters():
        param.requires_grad = False

    if scope == "text":
        for param in flow_lm.conditioner.embed.parameters():
            param.requires_grad = True
    elif scope == "text+flow":
        for param in flow_lm.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown train scope: {scope}")


def _flow_matching_loss(
    flow_lm,
    text_tokens: torch.Tensor,
    latents: torch.Tensor,
    audio_conditioning: torch.Tensor | None = None,
) -> torch.Tensor:
    bsz, seq_len, dim = latents.shape
    device = latents.device

    t = torch.rand((bsz, seq_len, 1), device=device)
    s = torch.zeros_like(t)
    x0 = torch.randn_like(latents)
    x1 = latents
    x_t = (1.0 - t) * x0 + t * x1
    u_t = x1 - x0

    text_embeddings = flow_lm.conditioner(TokenizedText(text_tokens))
    if audio_conditioning is not None:
        text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)
    input_ = flow_lm.input_linear(x_t)
    model_state = init_states(flow_lm, batch_size=bsz, sequence_length=seq_len + text_embeddings.shape[1])
    transformer_out = flow_lm.backbone(input_, text_embeddings, x_t, model_state=model_state)

    c = transformer_out.reshape(-1, transformer_out.shape[-1])
    x_flat = x_t.reshape(-1, dim)
    u_flat = u_t.reshape(-1, dim)
    s_flat = s.reshape(-1, 1)
    t_flat = t.reshape(-1, 1)

    pred = flow_lm.flow_net(c, s_flat, t_flat, x_flat)
    return F.mse_loss(pred, u_flat)


def _encode_audio_with_device_fix(model, audio: torch.Tensor, device: str) -> torch.Tensor:
    """Encode audio using Mimi with proper device-aware state initialization."""
    with torch.no_grad():
        from pocket_tts.models.mimi import pad_for_conv1d

        def init_and_move_state(module, seq_len):
            state = init_states(module, batch_size=1, sequence_length=seq_len)
            for module_state in state.values():
                for key, value in module_state.items():
                    if isinstance(value, torch.Tensor):
                        module_state[key] = value.to(device)
            return state

        # Initialize state for all stateful modules
        encoder_state = init_and_move_state(model.mimi.encoder, audio.shape[-1])
        transformer_state = init_and_move_state(model.mimi.encoder_transformer, audio.shape[-1])
        downsample_state = init_and_move_state(model.mimi.downsample, audio.shape[-1])

        # Combine all states
        mimi_state = {**encoder_state, **transformer_state, **downsample_state}

        # Temporarily replace methods to use our initialized state
        original_to_framerate = model.mimi._to_framerate
        original_encode = model.mimi.encode_to_latent

        def to_framerate_with_state(x, model_state=None):
            return model.mimi.downsample(x, model_state=mimi_state)

        def encode_with_state(x):
            frame_size = model.mimi.frame_size
            x_padded = pad_for_conv1d(x, frame_size, frame_size)
            emb = model.mimi.encoder(x_padded, model_state=mimi_state)
            (emb,) = model.mimi.encoder_transformer(emb, model_state=mimi_state)
            return model.mimi._to_framerate(emb)

        model.mimi._to_framerate = to_framerate_with_state
        model.mimi.encode_to_latent = encode_with_state
        try:
            latents = model.mimi.encode_to_latent(audio)
        finally:
            model.mimi.encode_to_latent = original_encode
            model.mimi._to_framerate = original_to_framerate
        latents = latents.transpose(-1, -2).to(torch.float32)
    return latents


def _prepare_voice_conditioning(model, voice_path: Path, device: str) -> torch.Tensor:
    # Re-encode the prompt to retrieve Mimi latents (detached).
    audio, _ = sf.read(voice_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = torch.from_numpy(audio)[None, None, :].to(device)
    latents = _encode_audio_with_device_fix(model, audio, device)
    return latents.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/processed/metadata.jsonl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--train-scope", choices=["text", "text+flow"], default="text+flow")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-audio-seconds", type=float, default=6.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--voice-prompt", default="samples/reference.wav")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--latents-metadata", default=None, help="Use precomputed latents metadata (from extract_latents.py)")
    parser.add_argument("--use-precomputed-latents", action="store_true", help="Load latents from latent_path in metadata")
    parser.add_argument("--use-lora", action="store_true", help="Apply LoRA adapters to flow LM linear layers (Unsloth-style)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = TTSModel.load_model()
    model.to(device)
    model.mimi.eval()

    if args.use_lora:
        _replace_linear_with_lora(model.flow_lm, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # audio_proj projects mimi latents (512) to flow_lm input dim (ldim)
    # This is for the flow matching loss, separate from voice conditioning
    audio_proj = torch.nn.Linear(512, model.flow_lm.ldim).to(device)

    # CRITICAL: Store audio_proj in the model so it can be used during generation
    # This is the key connection between training and generation!
    model.audio_proj = audio_proj
    voice_prompt = Path(args.voice_prompt)
    if not voice_prompt.exists():
        raise FileNotFoundError(f"Voice prompt not found: {voice_prompt}")
    voice_prompt_latents = _prepare_voice_conditioning(model, voice_prompt, device)

    # Also train speaker_proj_weight for voice conditioning
    # Make it a trainable parameter
    model.flow_lm.speaker_proj_weight.requires_grad = True

    _set_train_scope(model.flow_lm, args.train_scope)
    optimizer = AdamW(
        [
            {"params": filter(lambda p: p.requires_grad, model.flow_lm.parameters())},
            {"params": audio_proj.parameters()},
        ],
        lr=args.lr,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.max_steps, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device == "cuda")

    step = 0
    start_epoch = 0
    best_val = float("inf")
    bad_epochs = 0
    if args.resume and checkpoint_dir.exists():
        ckpts = sorted(checkpoint_dir.glob("flow_lm_step_*.pt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            latest = ckpts[-1]
            try:
                state = torch.load(latest, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(latest, map_location=device)
            model.flow_lm.load_state_dict(state["flow_lm"], strict=False)
            audio_proj.load_state_dict(state["audio_proj"], strict=False)
            if "speaker_proj_weight" in state:
                with torch.no_grad():
                    model.flow_lm.speaker_proj_weight.copy_(state["speaker_proj_weight"])
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            if "scaler" in state and args.fp16 and device == "cuda":
                scaler.load_state_dict(state["scaler"])
            step = int(state.get("step", 0))
            start_epoch = int(state.get("epoch", 0))

    metadata_source = Path(args.latents_metadata) if args.latents_metadata else metadata_path
    dataset = TTSDataset(metadata_source, use_latents=args.use_precomputed_latents)
    train_idx, val_idx = _split_indices(len(dataset), args.val_ratio, args.seed)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    for epoch in range(start_epoch, args.epochs):
        iterator = tqdm(loader, desc=f"epoch {epoch+1}")
        for batch in iterator:
            if step >= args.max_steps:
                break

            texts = batch["text_romanized"]
            if isinstance(texts, str):
                texts = [texts]
            audio_paths = batch["audio_path"]
            sample_rates = batch.get("sample_rate", 24000)
            if torch.is_tensor(sample_rates):
                if sample_rates.numel() == 1:
                    sample_rates = int(sample_rates.item())
                else:
                    sample_rates = [int(x) for x in sample_rates.tolist()]

            losses = []
            for idx, text in enumerate(texts):
                if not text:
                    text = simple_latin_transliterate(normalize_arabic(batch["text"][idx]))
                audio_path = Path(audio_paths[idx])
                if args.use_precomputed_latents and "latent_path" in batch:
                    lat_path = Path(batch["latent_path"][idx] if isinstance(batch["latent_path"], (list, tuple)) else batch["latent_path"])
                    latent_tensor = torch.load(lat_path)["latents"]
                    latents = latent_tensor.to(device)
                else:
                    if isinstance(sample_rates, (list, tuple)):
                        sr = int(sample_rates[idx])
                    else:
                        sr = int(sample_rates)
                    audio = _load_audio(audio_path, target_sr=sr, max_seconds=args.max_audio_seconds)
                    audio = audio.to(device, non_blocking=True)[None, None, :]

                    with torch.no_grad():
                        latents = _encode_audio_with_device_fix(model, audio, device)

                latents = audio_proj(latents)

                tokens = model.flow_lm.conditioner.prepare(text).tokens.to(device)
                tokens = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens

                conditioning = torch.nn.functional.linear(
                    voice_prompt_latents, model.flow_lm.speaker_proj_weight
                )
                if conditioning.shape[0] != latents.shape[0]:
                    conditioning = conditioning.expand(latents.shape[0], -1, -1)
                autocast_enabled = args.fp16 and device == "cuda"
                with torch.amp.autocast("cuda", enabled=autocast_enabled):
                    losses.append(
                        _flow_matching_loss(
                            model.flow_lm, tokens, latents, audio_conditioning=conditioning
                        )
                    )

            loss = torch.stack(losses).mean() / max(args.grad_accum, 1)
            if args.fp16 and device == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum == 0:
                if args.fp16 and device == "cuda":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.flow_lm.parameters(), max_norm=1.0)
                if args.fp16 and device == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            if step > 0 and step % args.save_every == 0:
                ckpt_path = checkpoint_dir / f"flow_lm_step_{step}.pt"
                torch.save(
                    {
                        "flow_lm": model.flow_lm.state_dict(),
                        "audio_proj": audio_proj.state_dict(),
                        "speaker_proj_weight": model.flow_lm.speaker_proj_weight,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if args.fp16 and device == "cuda" else None,
                        "step": step,
                        "epoch": epoch,
                    },
                    ckpt_path,
                )

            step += 1

            if step > 0 and step % args.eval_every == 0:
                model.flow_lm.eval()
                audio_proj.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_text = val_batch["text_romanized"][0]
                        if not val_text:
                            val_text = simple_latin_transliterate(
                                normalize_arabic(val_batch["text"][0])
                            )
                        val_audio = _load_audio(
                            Path(val_batch["audio_path"][0]),
                            target_sr=int(val_batch["sample_rate"][0]),
                            max_seconds=args.max_audio_seconds,
                        )
                        val_audio = val_audio.to(device, non_blocking=True)[None, None, :]
                        val_latents = _encode_audio_with_device_fix(model, val_audio, device)
                        val_latents = audio_proj(val_latents)
                        val_tokens = model.flow_lm.conditioner.prepare(val_text).tokens.to(device)
                        val_tokens = (
                            val_tokens.unsqueeze(0) if val_tokens.dim() == 1 else val_tokens
                        )
                        val_conditioning = torch.nn.functional.linear(
                            voice_prompt_latents, model.flow_lm.speaker_proj_weight
                        )
                        val_loss = _flow_matching_loss(
                            model.flow_lm,
                            val_tokens,
                            val_latents,
                            audio_conditioning=val_conditioning,
                        )
                        val_losses.append(val_loss.item())
                avg_val = float(np.mean(val_losses)) if val_losses else float("inf")
                model.flow_lm.train()
                audio_proj.train()
                if avg_val < best_val:
                    best_val = avg_val
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= args.early_stop_patience:
                    print(f"Early stopping triggered at step {step} (val {avg_val:.4f}).")
                    step = args.max_steps
                    break

        if step >= args.max_steps:
            break

    final_path = checkpoint_dir / "flow_lm_final.pt"
    torch.save(
        {
            "flow_lm": model.flow_lm.state_dict(),
            "audio_proj": audio_proj.state_dict(),
            "speaker_proj_weight": model.flow_lm.speaker_proj_weight,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if args.fp16 and device == "cuda" else None,
            "step": step,
            "epoch": args.epochs,
        },
        final_path,
    )
    print(f"Saved checkpoint to {final_path}")


if __name__ == "__main__":
    main()
