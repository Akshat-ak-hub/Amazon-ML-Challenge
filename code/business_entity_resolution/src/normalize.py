"""Stage 3 — Normalization: shared, source-agnostic cleaning.

Applied identically to Source 1, 2, and 3 so nothing is source-specific. Turns
noisy name/address strings into comparable, decomposed parts.

Handles the real noise seen in the data:
  - Devanagari / non-ASCII transliteration (unidecode folds to ASCII)
  - legal suffixes (Corp/Pvt/Ltd/LLC/Inc/GmbH/SARL/SAS ...) split off
  - punctuation & junk prefixes ('--', '<<', '&' vs 'and')
  - address abbreviations (Rd/Road, St/Street), landmarks ('Near SBI ATM')
  - missing components (empty address kept as null, never imputed)

Country is kept as a raw open-set string; the only derived signal is a
same/different/unknown flag (computed later in features), never one-hot — this is
what lets France (test-only) flow through unchanged.

Pure-Python Soundex (no native build deps) provides a phonetic code for names.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from unidecode import unidecode

# ---------------------------------------------------------------------------
# Legal-suffix vocabulary (multi-region). Matched as whole tokens, removed from
# the "core name" but recorded separately so suffix inconsistency doesn't tank
# similarity. Values map surface forms -> canonical tag.
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = {
    # English / US
    "corp": "corp", "corporation": "corp", "inc": "inc", "incorporated": "inc",
    "llc": "llc", "llp": "llp", "ltd": "ltd", "limited": "ltd", "co": "co",
    "company": "co", "plc": "plc", "lp": "lp",
    # India
    "pvt": "pvt", "private": "pvt", "pl": "pvt",
    # France / EU
    "sarl": "sarl", "sas": "sas", "sa": "sa", "sci": "sci", "eurl": "eurl",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
}

# Common address abbreviation standardization (expand -> canonical).
ADDR_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard", "ln": "lane", "dr": "drive",
    "ct": "court", "sq": "square", "hwy": "highway", "pkwy": "parkway",
    "apt": "apartment", "ste": "suite", "fl": "floor", "bldg": "building",
    "no": "number", "ph": "phase", "opp": "opposite", "nr": "near",
    "kh": "khasra", "gali": "lane", "marg": "road", "rue": "rue", "r": "rue",
    "blv": "boulevard", "bd": "boulevard",
}

# Landmark cue words (Indian-style "Near/Opp/Behind <X>") to isolate.
LANDMARK_CUES = {"near", "opp", "opposite", "behind", "beside", "next", "adjacent",
                 "above", "below", "front"}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_MULTISPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Phonetic: pure-Python Soundex
# ---------------------------------------------------------------------------
_SOUNDEX_MAP = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    **dict.fromkeys("l", "4"),
    **dict.fromkeys("mn", "5"),
    **dict.fromkeys("r", "6"),
}


def soundex(word: str) -> str:
    """Classic Soundex code (letter + 3 digits). Empty string for empty input."""
    word = re.sub(r"[^a-z]", "", unidecode(word).lower())
    if not word:
        return ""
    first = word[0]
    encoded = [_SOUNDEX_MAP.get(c, "") for c in word]
    # collapse adjacent duplicates
    out = [first.upper()]
    prev = _SOUNDEX_MAP.get(first, "")
    for c, code in zip(word[1:], encoded[1:]):
        if code and code != prev:
            out.append(code)
        if c not in "hw":  # h,w don't reset the "previous" per classic rules
            prev = code
    digits = "".join(out[1:])[:3]
    return (out[0] + digits + "000")[:4]


# ---------------------------------------------------------------------------
# Normalization results
# ---------------------------------------------------------------------------
@dataclass
class NormName:
    raw: str
    full: str                 # normalized full name (suffix retained)
    core: str                 # core name (legal suffix removed)
    tokens: list = field(default_factory=list)
    legal: list = field(default_factory=list)   # canonical legal tags
    phonetic: str = ""        # soundex of core-name tokens joined


@dataclass
class NormAddr:
    raw: str
    full: str                 # normalized full address string ("" if missing)
    tokens: list = field(default_factory=list)
    landmark: str = ""        # isolated landmark phrase ("" if none)
    pin: str = ""             # PIN/ZIP if detectable ("" otherwise)
    present: bool = False      # False when address is blank


def _basic_clean(text: str) -> str:
    """Lowercase, transliterate to ASCII, strip punctuation, collapse spaces."""
    if not text:
        return ""
    t = unidecode(text)
    t = t.lower()
    t = t.replace("&", " and ")
    t = _PUNCT_RE.sub(" ", t)
    t = _MULTISPACE_RE.sub(" ", t).strip()
    return t


def normalize_name(raw: str) -> NormName:
    cleaned = _basic_clean(raw or "")
    tokens = cleaned.split()

    # Merge runs of single letters (from dotted acronyms like "S.A.S" -> s a s)
    # into one token when the joined form is a known legal suffix. This catches
    # French/EU acronym suffixes that punctuation-stripping fragments.
    merged = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1 and tokens[i].isalpha():
            j = i
            run = []
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                run.append(tokens[j])
                j += 1
            joined = "".join(run)
            if len(run) >= 2 and joined in LEGAL_SUFFIXES:
                merged.append(joined)
                i = j
                continue
            # not a suffix acronym: keep letters as-is
            merged.extend(run)
            i = j
        else:
            merged.append(tokens[i])
            i += 1
    tokens = merged

    legal = []
    core_tokens = []
    for tok in tokens:
        if tok in LEGAL_SUFFIXES:
            legal.append(LEGAL_SUFFIXES[tok])
        else:
            core_tokens.append(tok)
    core = " ".join(core_tokens)
    phonetic = " ".join(soundex(t) for t in core_tokens if t)
    return NormName(
        raw=raw or "",
        full=cleaned,
        core=core,
        tokens=core_tokens,
        legal=sorted(set(legal)),
        phonetic=phonetic,
    )


_PIN_RE = re.compile(r"\b(\d{5,6})\b")            # US ZIP (5) / India PIN (6)


def normalize_address(raw: str) -> NormAddr:
    if not raw or not raw.strip():
        return NormAddr(raw=raw or "", full="", tokens=[], present=False)

    # PIN/ZIP detection on the raw-ish text before punctuation stripping.
    pin = ""
    m = _PIN_RE.search(unidecode(raw))
    if m:
        pin = m.group(1)

    cleaned = _basic_clean(raw)
    tokens = cleaned.split()

    # Landmark isolation: capture "<cue> <up to 3 words>".
    landmark_parts = []
    kept = []
    i = 0
    while i < len(tokens):
        if tokens[i] in LANDMARK_CUES:
            landmark_parts.append(tokens[i])
            j = i + 1
            taken = 0
            while j < len(tokens) and taken < 3:
                landmark_parts.append(tokens[j])
                j += 1
                taken += 1
            i = j
        else:
            kept.append(tokens[i])
            i += 1

    # Abbreviation standardization on the non-landmark tokens.
    std = [ADDR_ABBREV.get(t, t) for t in kept]

    return NormAddr(
        raw=raw,
        full=" ".join(std),
        tokens=std,
        landmark=" ".join(landmark_parts),
        pin=pin,
        present=True,
    )


def normalize_country(raw: str) -> str:
    """Open-set: just trimmed, case-folded string. Never a fixed vocabulary."""
    return (raw or "").strip().lower()
