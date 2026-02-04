import re


_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"

_NORMALIZATION_MAP = {
    "\u0622": "\u0627",  # ALEF WITH MADDA ABOVE -> ALEF
    "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE -> ALEF
    "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW -> ALEF
    "\u0671": "\u0627",  # ALEF WASLA -> ALEF
    "\u0649": "\u064A",  # ALEF MAKSURA -> YEH
    "\u0624": "\u0648",  # WAW WITH HAMZA -> WAW
    "\u0626": "\u064A",  # YEH WITH HAMZA -> YEH
}

_BUCKWALTER_MAP = {
    "ا": "A",
    "أ": ">", "إ": "<", "آ": "|",
    "ب": "b",
    "ت": "t",
    "ث": "v",
    "ج": "j",
    "ح": "H",
    "خ": "x",
    "د": "d",
    "ذ": "*",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "$",
    "ص": "S",
    "ض": "D",
    "ط": "T",
    "ظ": "Z",
    "ع": "E",
    "غ": "g",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",
    "ي": "y",
    "ء": "'",
    "ؤ": "&",
    "ئ": "}",
    "ة": "p",
    "ى": "Y",
    "\u064B": "F",  # FATHATAN
    "\u064C": "N",  # DAMMATAN
    "\u064D": "K",  # KASRATAN
    "\u064E": "a",  # FATHA
    "\u064F": "u",  # DAMMA
    "\u0650": "i",  # KASRA
    "\u0651": "~",  # SHADDA
    "\u0652": "o",  # SUKUN
}


_LATIN_MAP = {
    "ا": "a",
    "ب": "b",
    "ت": "t",
    "ث": "th",
    "ج": "j",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ذ": "dh",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "d",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "و": "w",
    "ي": "y",
    "ء": "a",
    "ة": "a",
}

_EXTRA_LATIN_MAP = {
    "\u064E": "a",  # FATHA
    "\u064F": "u",  # DAMMA
    "\u0650": "i",  # KASRA
    "\u064B": "an",  # FATHATAN (tanween)
    "\u064C": "un",  # DAMMATAN (tanween)
    "\u064D": "in",  # KASRATAN (tanween)
    "\u0651": "2",  # SHADDA (gemination)
    "\u0652": "",  # SUKUN - removed for cleaner Arabizi (no vowel)
    "\u0627": "a",  # ALEF -> 'a' for more natural Arabizi (was 'A')
    "\u0648": "u",  # WAW as long vowel 'u' (when part of vowel)
    "\u064A": "i",  # YEH as long vowel 'i' (when part of vowel)
    "\u0640": "",  # TATWEEL - removed
}

# Updating specific keys for Arabizi style (best practice Arabizi mappings)
_LATIN_MAP.update({
    "ا": "a",      # alif -> 'a'
    "و": "w",      # waw -> 'w' (consonant)
    "ي": "y",      # yeh -> 'y' (consonant)
    "ء": "",       # hamza - typically omitted in Arabizi
    "أ": "",       # alif with hamza above - omit
    "إ": "",       # alif with hamza below - omit
    "آ": "aa",     # alif madda -> 'aa'
    "ة": "a",      # ta marbuta -> 'a'
    "ع": "3",      # ayn -> '3' (classic Arabizi)
    "غ": "gh",     # ghayn -> 'gh'
    "ح": "7",      # ha -> '7' (classic Arabizi) or use 'h' for simpler form
    "خ": "5",      # kha -> '5' (classic Arabizi)
    "ص": "s",      # sad -> 's' (more readable than 'S')
    "ض": "d",      # dad -> 'd' (more readable than 'D')
    "ط": "t",      # ta -> 't' (more readable than 'T')
    "ظ": "z",      # dha -> 'z' (more readable than 'Z')
    "ق": "q",      # qaf -> 'q'
    "ث": "th",     # tha -> 'th'
    "ذ": "dh",     # dhal -> 'dh'
    "ش": "sh",     # shin -> 'sh'
})
_LATIN_MAP.update(_EXTRA_LATIN_MAP)

# Phonetic map for TTS model input (no numbers, uses letters)
_PHONETIC_MAP = _LATIN_MAP.copy()
_PHONETIC_MAP.update({
    "ا": "a",      # alif -> 'a'
    "و": "w",      # waw -> 'w' (consonant)
    "ي": "y",      # yeh -> 'y' (consonant)
    "ء": "",       # hamza - omit
    "أ": "",       # alif with hamza above - omit
    "إ": "",       # alif with hamza below - omit
    "آ": "aa",     # alif madda -> 'aa'
    "ة": "a",      # ta marbuta -> 'a'
    "ع": "a",      # ayn -> 'a' (phonetic, not '3')
    "غ": "gh",     # ghayn -> 'gh'
    "ح": "h",      # ha -> 'h' (phonetic, not '7')
    "خ": "kh",     # kha -> 'kh' (phonetic, not '5')
    "ص": "s",      # sad -> 's'
    "ض": "d",      # dad -> 'd'
    "ط": "t",      # ta -> 't'
    "ظ": "z",      # dha -> 'z'
    "ق": "q",      # qaf -> 'q'
    "ث": "th",     # tha -> 'th'
    "ذ": "dh",     # dhal -> 'dh'
    "ش": "sh",     # shin -> 'sh'
})

# Arabizi map for display (uses numbers like 7, 3, 5)
_ARABIZI_MAP = _LATIN_MAP.copy()
_ARABIZI_MAP.update({
    "ا": "a",      # alif -> 'a'
    "و": "w",      # waw -> 'w' (consonant)
    "ي": "y",      # yeh -> 'y' (consonant)
    "ء": "",       # hamza - omit
    "أ": "",       # alif with hamza above - omit
    "إ": "",       # alif with hamza below - omit
    "آ": "aa",     # alif madda -> 'aa'
    "ة": "a",      # ta marbuta -> 'a'
    "ع": "3",      # ayn -> '3' (classic Arabizi)
    "غ": "gh",     # ghayn -> 'gh'
    "ح": "7",      # ha -> '7' (classic Arabizi)
    "خ": "5",      # kha -> '5' (classic Arabizi)
    "ص": "s",      # sad -> 's'
    "ض": "d",      # dad -> 'd'
    "ط": "t",      # ta -> 't'
    "ظ": "z",      # dha -> 'z'
    "ق": "q",      # qaf -> 'q'
    "ث": "th",     # tha -> 'th'
    "ذ": "dh",     # dhal -> 'dh'
    "ش": "sh",     # shin -> 'sh'
})

def normalize_arabic(text: str, keep_diacritics: bool = False) -> str:
    text = text.replace(_TATWEEL, "")
    if not keep_diacritics:
        text = _ARABIC_DIACRITICS.sub("", text)
    # We maintain the normalization map for consistency, but might want to be careful 
    # if it conflates letters we now want distinct.
    # For now, we apply normalization map to standardize shapes.
    return "".join(_NORMALIZATION_MAP.get(ch, ch) for ch in text)


def buckwalter_transliterate(text: str) -> str:
    return "".join(_BUCKWALTER_MAP.get(ch, ch) for ch in text)


def simple_latin_transliterate(text: str, keep_diacritics: bool = True, infer_vowels: bool = True,
                              use_arabizi_numbers: bool = True) -> str:
    """
    Latin transliteration (Arabizi-style) for readability and phonetic accuracy.
    Now supports diacritics (including tanween) for better vowel representation.

    Args:
        text: Arabic text to transliterate
        keep_diacritics: Whether to preserve Arabic diacritics
        infer_vowels: Whether to infer missing vowels for common patterns
        use_arabizi_numbers: Use number substitutions (7=ح, 3=ع, 5=خ). Set False for TTS input.
    """
    # Normalize but keep diacritics if requested
    text = normalize_arabic(text, keep_diacritics=keep_diacritics)

    # Choose the appropriate mapping based on use_arabizi_numbers flag
    char_map = _ARABIZI_MAP if use_arabizi_numbers else _PHONETIC_MAP

    # Tanween characters (alif/waw/yeh are just seats for tanween)
    TANWEEN = {"\u064B", "\u064C", "\u064D"}  # FATHATAN, DAMMATAN, KASRATAN
    # Long vowel seats that should be skipped with adjacent tanween
    TANWEEN_SEATS = {"ا", "و", "ي"}

    # First pass: transliterate with handling for tanween
    temp_out = []
    i = 0
    while i < len(text):
        ch = text[i]

        # Case 1: Seat followed by tanween (اً)
        if (i + 1 < len(text) and text[i + 1] in TANWEEN and
            ch in TANWEEN_SEATS):
            # Skip the seat, output only tanween
            temp_out.append(char_map.get(text[i + 1], text[i + 1]))
            i += 2
            continue

        # Case 2: Tanween followed by seat (ًا) - Unicode storage order
        if (ch in TANWEEN and i + 1 < len(text) and
            text[i + 1] in TANWEEN_SEATS):
            # Skip the seat, output only tanween
            temp_out.append(char_map.get(ch, ch))
            i += 2
            continue

        temp_out.append(char_map.get(ch, ch))
        i += 1

    result = "".join(temp_out)

    # Second pass: infer vowels for common patterns if requested
    if infer_vowels:
        result = _infer_vowels(result)

    return result


def _infer_vowels(text: str) -> str:
    """
    Infer missing vowels for common Arabic patterns to produce more natural Arabizi.
    This is a heuristic-based approach for common words and patterns.
    """
    import re

    result = text

    # Common word replacements (without diacritics → with inferred vowels)
    # Pronouns and particles
    replacements = [
        (r'\bbkm\b', 'bikum'),       # بكم → bikum
        (r'\bbk\b', 'bik'),          # بك → bik
        (r'\bfyh\b', 'fiyh'),        # فيه → fiyh
        (r'\bfy\b', 'fi'),           # في → fi
        (r'\b3ly\b', '3ali'),        # علي → 3ali
        (r'\b3l\b', '3al'),          # على → 3al
        (r'\bmn\b', 'min'),          # من → min
        (r'\bil\b', 'ila'),          # الى → ila/ilā
        (r'\bhnk\b', 'hinak'),       # هنك → hinak
        (r'\bh\b', 'ha'),            # ها → ha (when alone)
        (r'\bdhlk\b', 'dhalk'),      # ذلك → dhalk
        (r'\bhzh\b', 'hadh'),        # هذا → hadh
        (r'\bhthy\b', 'hathi'),      # هذه → hathi
    ]

    # Apply replacements
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # General patterns:
    # 1. After 'al' (ال), add 'i' after first consonant of root for many words
    #    alkitaab → alkitaab (no change needed), but alktb → alkutab or alkitaab
    #    This is harder to generalize, so we skip it for now

    # 2. Common bi-/tri-consonant patterns
    #    ktb → katab (to write), ktb → kutub (books)
    #    These require dictionary lookup, so we skip for now

    # 3. Vowel insertion for consonant clusters (simple heuristic)
    #    If 3+ consonants in a row without vowels, insert hints
    #    e.g., "ktb" → "katab" (not always correct, but better than nothing)

    return result
