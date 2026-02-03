import argparse
from pathlib import Path

import soundfile as sf
import torch

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645 \u0641\u064a \u0639\u0627\u0644\u0645 \u0627\u0644\u0645\u0639\u0644\u0648\u0645\u0627\u062a.")
    parser.add_argument("--output", default="samples/generated_arabic.wav")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--voice", default="samples/reference.wav")
    parser.add_argument("--romanize", action="store_true", help="Force Buckwalter-style transliteration.")
    parser.add_argument(
        "--latinize",
        action="store_true",
        help="Force simple Latin transliteration (chat-style with 3/7, matches training).",
    )
    parser.add_argument(
        "--no-auto-transliterate",
        action="store_true",
        help="Disable automatic transliteration fallback (default: auto-detect Arabic and latinize).",
    )
    parser.add_argument("--frames-after-eos", type=int, default=8)
    parser.add_argument("--eos-threshold", type=float, default=-6.0)
    parser.add_argument("--temp", type=float, default=None, help="Sampling temperature (lower = slower/clearer)")
    parser.add_argument(
        "--lsd-decode-steps",
        type=int,
        default=None,
        help="LSD decode steps (higher can slow pace slightly; default model setting if None)",
    )
    parser.add_argument("--normalize", action="store_true", help="Peak-normalize output to 0.95")
    args = parser.parse_args()

    load_kwargs = {"eos_threshold": args.eos_threshold}
    if args.temp is not None:
        load_kwargs["temp"] = args.temp
    if args.lsd_decode_steps is not None:
        load_kwargs["lsd_decode_steps"] = args.lsd_decode_steps
    model = TTSModel.load_model(**load_kwargs)

    if args.checkpoint:
        try:
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            # Torch <2.3 or checkpoints without safe globals
            state = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(state, dict):
            if "flow_lm" in state:
                model.flow_lm.load_state_dict(state["flow_lm"], strict=False)
            if "speaker_proj_weight" in state:
                # Load the trained speaker_proj_weight directly
                with torch.no_grad():
                    model.flow_lm.speaker_proj_weight.copy_(state["speaker_proj_weight"])
                print("Loaded trained speaker_proj_weight from checkpoint")
        else:
            model.flow_lm.load_state_dict(state, strict=False)

    voice_state = model.get_state_for_audio_prompt(args.voice)

    original_text = args.text
    text = original_text
    readable = buckwalter_transliterate(normalize_arabic(original_text))
    # Keep inference aligned with training: training data used simple_latin_transliterate.
    def is_arabic(s: str) -> bool:
        return any("\u0600" <= ch <= "\u06FF" for ch in s)

    if args.latinize:
        text = simple_latin_transliterate(text)
    elif args.romanize:
        text = buckwalter_transliterate(normalize_arabic(text))
    elif not args.no_auto_transliterate and is_arabic(text):
        text = simple_latin_transliterate(normalize_arabic(text))

    # Log the resolved text so it's visible in the terminal
    print(f"Input text: {original_text}")
    if text != original_text:
        print(f"Model text: {text}")
    print(f"Readable (buckwalter): {readable}")

    audio = model.generate_audio(
        voice_state, text, frames_after_eos=args.frames_after_eos, copy_state=True
    )
    if args.normalize:
        peak = audio.abs().max().item()
        if peak > 0:
            audio = audio / peak * 0.95
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio.squeeze().numpy(), model.sample_rate)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
