"""Unit tests for normalize.py — using REAL noisy examples seen in the dataset.

Run:  python -m pytest tests/test_normalize.py -q
  or: python tests/test_normalize.py   (falls back to a plain runner)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import normalize as N  # noqa: E402


def test_legal_suffix_split_us():
    n = N.normalize_name("B+ Retail Inc")
    assert "inc" in n.legal
    assert "inc" not in n.core.split()
    assert "retail" in n.tokens


def test_and_ampersand_equivalence():
    a = N.normalize_name("Smith & Sons")
    b = N.normalize_name("Smith and Sons")
    assert a.core == b.core


def test_corp_variants_canonicalize():
    assert N.normalize_name("Acme Robotics Incorporated").legal == ["inc"]
    assert N.normalize_name("Acme Robotics Inc").legal == ["inc"]
    assert N.normalize_name("Acme Corp").legal == ["corp"]
    assert N.normalize_name("Acme Corporation").legal == ["corp"]


def test_devanagari_transliteration():
    # राम मार्केटिंग प्राइवेट लिमिटेड  (Ram Marketing Private Limited)
    n = N.normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert n.core  # non-empty after transliteration
    assert n.full.isascii()  # folded to ASCII


def test_junk_prefix_stripped():
    n = N.normalize_name("-- Holloway Peak Inc Seafood")
    assert "--" not in n.core
    assert "holloway" in n.tokens
    assert "inc" in n.legal


def test_double_angle_prefix():
    n = N.normalize_name("<< Team Ecole")
    assert n.core.strip().startswith("team")


def test_french_legal_suffix():
    n = N.normalize_name("Fractales Amis Groupe S.A.S")
    assert "sas" in n.legal


def test_empty_address_marked_absent():
    a = N.normalize_address("")
    assert a.present is False
    assert a.full == ""


def test_address_abbrev_standardized():
    a = N.normalize_address("Mack Rd, Haltom City, Texas")
    assert "road" in a.tokens
    assert "rd" not in a.tokens


def test_pin_detection_india():
    a = N.normalize_address("G-3/571, Gulmohar Colony, Bhopal 462001, Madhya Pradesh")
    assert a.pin == "462001"


def test_zip_detection_us():
    a = N.normalize_address("1795 Westchester Drive, High Point, NC 27262")
    assert a.pin == "27262"


def test_landmark_isolation():
    a = N.normalize_address("Shop 4, Near SBI ATM, MG Road")
    assert "near" in a.landmark
    assert "sbi" in a.landmark
    # landmark words removed from main tokens
    assert "sbi" not in a.tokens


def test_soundex_typo_robustness():
    # typos should collapse to the same phonetic code
    assert N.soundex("Robert") == N.soundex("Rupert")
    assert N.soundex("Learning") == N.soundex("Léarning")  # accent variant


def test_country_openset():
    assert N.normalize_country(" France ") == "france"
    assert N.normalize_country("US") == "us"
    # no fixed vocabulary — an unseen label passes through
    assert N.normalize_country("Germany") == "germany"


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
