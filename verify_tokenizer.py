"""
Quick script to verify that the tokenizer recognizes simple_latin_transliterate
better than buckwalter_transliterate for Arabic text.
"""
import sys
import io

# Force UTF-8 output for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from arabic_utils import buckwalter_transliterate, simple_latin_transliterate, normalize_arabic

try:
    from pocket_tts import TTSModel
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    local_repo = Path(__file__).parent / "_pocket-tts"
    if local_repo.exists():
        sys.path.insert(0, str(local_repo))
    from pocket_tts import TTSModel


def test_tokenizer(text: str, transliteration_method: str, model):
    """Test how many tokens are recognized vs unknown."""
    normalized = normalize_arabic(text)
    if transliteration_method == "buckwalter":
        romanized = buckwalter_transliterate(normalized)
    else:
        romanized = simple_latin_transliterate(normalized)

    tokens = model.flow_lm.conditioner.prepare(romanized).tokens
    unknown_count = (tokens == 3).sum().item()  # 3 is typically <unk> token
    total_count = tokens.numel()

    return {
        "original": text,
        "normalized": normalized,
        "romanized": romanized,
        "unknown_tokens": unknown_count,
        "total_tokens": total_count,
        "recognition_rate": (total_count - unknown_count) / total_count * 100 if total_count > 0 else 0
    }


def main():
    print("Loading Pocket TTS model...")
    model = TTSModel.load_model()

    # Test cases covering common Arabic text
    test_texts = [
        "السلام عليكم",  # Common greeting
        "بسم الله الرحمن الرحيم",  # Bismillah
        "الحمد لله رب العالمين",  # Alhamdulillah
        "لِأَنَّهُ لَا يَرَى أَنَّهُ عَلَى السَّفَهِ",  # From dataset
    ]

    print("\n" + "="*80)
    print("TOKENIZER RECOGNITION TEST")
    print("="*80)

    for text in test_texts:
        print(f"\nTesting: {text}")
        print("-" * 80)

        buckwalter_result = test_tokenizer(text, "buckwalter", model)
        print(f"Buckwalter: '{buckwalter_result['romanized']}'")
        print(f"  Tokens: {buckwalter_result['total_tokens']} total, {buckwalter_result['unknown_tokens']} unknown")
        print(f"  Recognition: {buckwalter_result['recognition_rate']:.1f}%")

        simple_result = test_tokenizer(text, "simple_latin", model)
        print(f"Simple Latin: '{simple_result['romanized']}'")
        print(f"  Tokens: {simple_result['total_tokens']} total, {simple_result['unknown_tokens']} unknown")
        print(f"  Recognition: {simple_result['recognition_rate']:.1f}%")

        improvement = simple_result['recognition_rate'] - buckwalter_result['recognition_rate']
        print(f"  Improvement: +{improvement:.1f} percentage points")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("If Simple Latin shows >90% recognition, the fix should work.")
    print("Regenerate your dataset with: python data_prep.py --dataset MBZUAI/ClArTTS --max-samples 50")


if __name__ == "__main__":
    main()
