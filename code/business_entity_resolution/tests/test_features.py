import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import normalize as N
import features as F


def feats(n1, a1, n2, a2):
    v = F.pair_features(N.normalize_name(n1), N.normalize_address(a1),
                        N.normalize_name(n2), N.normalize_address(a2))
    return dict(zip(F.FEATURE_NAMES, v))


def test_feature_count():
    v = feats("Acme Inc", "1 Main St", "Acme Inc", "1 Main St")
    assert len(v) == len(F.FEATURE_NAMES) == 19


def test_true_match_scores_high():
    v = feats("Acme Robotics Inc", "100 Main Street, Phoenix, AZ",
              "Acme Robotics Incorporated", "100 Main St, Phoenix AZ")
    assert v["name_exact_core"] == 1.0        # legal suffix stripped -> same core
    assert v["name_lev_ratio"] > 0.9
    assert v["addr_tok_jaccard"] > 0.5


def test_typo_still_similar():
    v = feats("Clemons Silver Eastern Inc", "1619 Julia Park Drive, Spring, TX",
              "Clemons Silvre Eastern Inc", "Julia Park Drive, Spring, Texas")
    assert v["name_lev_ratio"] > 0.8
    assert v["name_token_set"] > 0.7
    assert v["addr_tok_jaccard"] > 0.4


def test_word_order_swap():
    v = feats("Physical Therapy Associates Group", "x",
              "Group Physical Therapy Associates", "x")
    assert v["name_token_sort"] > 0.95        # order-robust should be ~1


def test_non_match_scores_low():
    v = feats("Lyrelle Wave LLC", "7503 Laytonia Drive, Gaithersburg, MD",
              "Avinex", "7503 Laytonia Drive, Gaithersburg, MD")
    assert v["name_lev_ratio"] < 0.5          # names totally different
    assert v["addr_tok_jaccard"] > 0.8        # but address identical -> useful signal


def test_country_flag():
    assert F.country_flag("US", "US") == 1.0
    assert F.country_flag("US", "India") == -1.0
    assert F.country_flag("", "France") == 0.0


def test_missing_address_flag():
    v = feats("Some Co", "", "Some Co", "123 Main St")
    assert v["addr_present_both"] == 0.0
    assert v["addr_tok_jaccard"] == 0.0


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    p = 0
    for fn in fns:
        try:
            fn(); p += 1; print("PASS", fn.__name__)
        except AssertionError as e:
            print("FAIL", fn.__name__, e)
        except Exception as e:
            print("ERROR", fn.__name__, type(e).__name__, e)
    print(f"\n{p}/{len(fns)} passed")
    return p == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
