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
    "\u0651": "2",  # SHADDA
    "\u0652": "o",  # SUKUN
    "\u0627": "A",  # ALEF -> A (Long vowel)
    "\u0640": "_",  # TATWEEL
}

# Updating specific keys for Arabizi style
_LATIN_MAP.update({
    "ا": "A", 
    "و": "w", 
    "ي": "y", 
    "ء": "2", 
    "أ": "2", 
    "إ": "2", 
    "آ": "2", 
    "ة": "a", 
    "ع": "3", 
    "غ": "gh",
    "ح": "7", 
    "خ": "5", 
    "ص": "S", 
    "ض": "D", 
    "ط": "T", 
    "ظ": "Z", 
    "ق": "q", 
})
_LATIN_MAP.update(_EXTRA_LATIN_MAP)

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


def simple_latin_transliterate(text: str, keep_diacritics: bool = True) -> str:
    """
    Latin transliteration (Arabizi-style) for readability and phonetic accuracy.
    Now supports diacritics (including tanween) for better vowel representation.
    """
    # Normalize but keep diacritics if requested
    text = normalize_arabic(text, keep_diacritics=keep_diacritics)
    
    out = []
    for ch in text:
        out.append(_LATIN_MAP.get(ch, ch))
    return "".join(out)
