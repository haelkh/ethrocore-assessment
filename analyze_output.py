from arabic_utils import buckwalter_transliterate, normalize_arabic, simple_latin_transliterate
import soundfile as sf

text = "مرحبا بكم في تجربة جديدة بعد استئناف التدريب. نريد صوتا عربيا واضحا وطويلا."
romanized = buckwalter_transliterate(normalize_arabic(text))
latinized = simple_latin_transliterate(text)
print(f"Original: {text}")
print(f"Buckwalter (Old Method): {romanized}")
print(f"Latinized (New Method):  {latinized}")

try:
    info = sf.info('samples/generated_arabic_latinized.wav')
    print(f"File: samples/generated_arabic_latinized.wav")
    print(f"Duration: {info.duration:.2f} seconds")
    print(f"Sample Rate: {info.samplerate}")
    print(f"Channels: {info.channels}")
except Exception as e:
    print(f"Could not analyze audio file: {e}")
