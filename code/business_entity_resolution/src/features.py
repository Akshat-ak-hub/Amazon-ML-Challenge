"""Stage 5a — Feature engineering: turn each (S1, candidate) pair into a vector.

For every candidate pair produced by blocking, compute a fixed set of numeric
similarity features over the normalized name + address, plus a country flag and
structural signals. These vectors feed the LightGBM matching model.

Feature groups:
  NAME
    - token Jaccard (core name)
    - Levenshtein ratio (rapidfuzz, normalized 0..1)
    - Jaro-Winkler
    - token_sort_ratio (word-order robust)
    - token_set_ratio
    - char-trigram Jaccard
    - phonetic-code overlap (Soundex tokens)
    - exact core-name match flag
    - full-vs-core similarity delta (isolates legal-suffix effect)
    - length ratio, token-count delta
  ADDRESS
    - present-on-both flag
    - token Jaccard
    - token_sort_ratio
    - char-4gram Jaccard
    - PIN present-both flag + PIN exact match
    - shared numeric tokens (house numbers) count
  COUNTRY
    - same / different / unknown (encoded 1 / -1 / 0)
  STRUCTURAL
    - (added by caller) shared blocking-key count / candidate rank

Memory: we hold id -> (name, addr) for all records referenced. To stay bounded we
load only the records that actually appear (S1 entities + their candidates), read
in a streaming pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

from rapidfuzz import fuzz, distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import normalize as N  # noqa: E402

DELIM = "\t"

FEATURE_NAMES = [
    "name_tok_jaccard", "name_lev_ratio", "name_jaro_winkler",
    "name_token_sort", "name_token_set", "name_trigram_jaccard",
    "name_phonetic_overlap", "name_exact_core", "name_full_core_delta",
    "name_len_ratio", "name_tokcount_delta",
    "addr_present_both", "addr_tok_jaccard", "addr_token_sort",
    "addr_4gram_jaccard", "addr_pin_both", "addr_pin_exact",
    "addr_shared_nums",
    "country_flag",
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ngrams(s: str, n: int) -> set:
    t = s.replace(" ", "")
    if len(t) <= n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def pair_features(nm1: N.NormName, ad1: N.NormAddr,
                  nm2: N.NormName, ad2: N.NormAddr) -> list:
    """Compute the feature vector for one (record1, record2) pair."""
    c1, c2 = nm1.core, nm2.core
    t1, t2 = set(nm1.tokens), set(nm2.tokens)

    # NAME
    name_tok_jaccard = _jaccard(t1, t2)
    name_lev_ratio = distance.Levenshtein.normalized_similarity(c1, c2) if (c1 or c2) else 1.0
    name_jaro_winkler = distance.JaroWinkler.normalized_similarity(c1, c2) if (c1 or c2) else 1.0
    name_token_sort = fuzz.token_sort_ratio(c1, c2) / 100.0
    name_token_set = fuzz.token_set_ratio(c1, c2) / 100.0
    name_trigram_jaccard = _jaccard(_ngrams(c1, 3), _ngrams(c2, 3))
    p1, p2 = set(nm1.phonetic.split()), set(nm2.phonetic.split())
    name_phonetic_overlap = _jaccard(p1, p2)
    name_exact_core = 1.0 if (c1 and c1 == c2) else 0.0
    full_sim = fuzz.ratio(nm1.full, nm2.full) / 100.0
    core_sim = fuzz.ratio(c1, c2) / 100.0
    name_full_core_delta = abs(full_sim - core_sim)
    l1, l2 = len(c1), len(c2)
    name_len_ratio = min(l1, l2) / max(l1, l2) if max(l1, l2) else 1.0
    name_tokcount_delta = abs(len(nm1.tokens) - len(nm2.tokens))

    # ADDRESS
    addr_present_both = 1.0 if (ad1.present and ad2.present) else 0.0
    if ad1.present and ad2.present:
        at1, at2 = set(ad1.tokens), set(ad2.tokens)
        addr_tok_jaccard = _jaccard(at1, at2)
        addr_token_sort = fuzz.token_sort_ratio(ad1.full, ad2.full) / 100.0
        addr_4gram_jaccard = _jaccard(_ngrams(ad1.full, 4), _ngrams(ad2.full, 4))
        nums1 = {t for t in ad1.tokens if t.isdigit()}
        nums2 = {t for t in ad2.tokens if t.isdigit()}
        addr_shared_nums = float(len(nums1 & nums2))
    else:
        addr_tok_jaccard = addr_token_sort = addr_4gram_jaccard = 0.0
        addr_shared_nums = 0.0
    addr_pin_both = 1.0 if (ad1.pin and ad2.pin) else 0.0
    addr_pin_exact = 1.0 if (ad1.pin and ad1.pin == ad2.pin) else 0.0

    # COUNTRY (same=1, different=-1, unknown=0) — computed from raw normalized cc
    # passed via nm objects? country isn't in NormName; caller supplies via wrapper.
    # Placeholder 0; overwritten by build step which knows the countries.
    country_flag = 0.0

    return [
        name_tok_jaccard, name_lev_ratio, name_jaro_winkler,
        name_token_sort, name_token_set, name_trigram_jaccard,
        name_phonetic_overlap, name_exact_core, name_full_core_delta,
        name_len_ratio, float(name_tokcount_delta),
        addr_present_both, addr_tok_jaccard, addr_token_sort,
        addr_4gram_jaccard, addr_pin_both, addr_pin_exact, addr_shared_nums,
        country_flag,
    ]


def country_flag(cc1: str, cc2: str) -> float:
    a, b = (cc1 or "").strip().lower(), (cc2 or "").strip().lower()
    if not a or not b:
        return 0.0
    return 1.0 if a == b else -1.0
