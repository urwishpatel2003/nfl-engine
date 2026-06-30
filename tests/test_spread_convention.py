"""
tests/test_spread_convention.py
-------------------------------
Regression guard for the nflverse spread_line sign convention.

The whole ATS/picks layer once had this backwards, which made backtests report
fake ~86% cover rates. These tests pin the convention down so it can't silently
flip again.

CONVENTION (verified empirically against schedules.parquet):
    spread_line > 0  ==  HOME team favored (home expected to win by spread_line)
    Home covers ATS  iff  (home_score - away_score) > spread_line

Run:  python -m pytest tests/ -q       (or)   python tests/test_spread_convention.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"


def _load_completed_games():
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[
        (s["game_type"].str.upper() == "REG")
        & s["home_score"].notna()
        & s["spread_line"].notna()
    ].copy()
    s["home_margin"] = s["home_score"] - s["away_score"]
    return s


def test_positive_spread_means_home_favored():
    """spread_line > 0 should correspond to home winning most of the time."""
    s = _load_completed_games()
    home_fav = s[s["spread_line"] > 0]
    away_fav = s[s["spread_line"] < 0]
    assert (home_fav["home_margin"] > 0).mean() > 0.60, "home favorites should win >60%"
    assert (away_fav["home_margin"] > 0).mean() < 0.45, "home underdogs should win <45%"


def test_home_cover_rule_is_roughly_fair():
    """Home covers iff home_margin > spread_line; the market should be ~50/50."""
    s = _load_completed_games()
    home_cover = s["home_margin"] > s["spread_line"]
    rate = home_cover.mean()
    assert 0.45 < rate < 0.55, f"home ATS cover rate {rate:.3f} not near 50% — sign likely flipped"


def test_spread_line_correlates_positively_with_margin():
    s = _load_completed_games()
    assert s["spread_line"].corr(s["home_margin"]) > 0.3, "spread_line should track home margin"


def test_backtest_picks_cover_logic_matches_convention():
    """score_pick's covered flag must agree with the canonical cover rule."""
    from backtest_picks import score_pick

    # score_pick only fires on disagreement, so set Vegas to favor AWAY (spread_line<0)
    # while the model favors home. spread_line=-3 => home is a +3 underdog.
    # Pick is HOME +3, which covers iff home_margin > -3.
    pred = {"home_team": "AAA", "away_team": "BBB",
            "predicted_home_score": 27, "predicted_away_score": 17,  # model: home by 10
            "home_win_probability": 0.7}

    # Home loses by 1 (margin -1 > -3) -> HOME +3 covers.
    game_cov = pd.Series({"spread_line": -3.0, "home_score": 23, "away_score": 24,
                          "home_rest": 7, "away_rest": 7, "roof": "outdoors"})
    out = score_pick(pred, game_cov, week=5)
    assert out is not None and out["pick_side"] == "home", "model+home vs vegas+away should pick home"
    assert out["covered"] is True, "home +3 losing by 1 must cover"

    # Home loses by 5 (margin -5 > -3 is False) -> HOME +3 does NOT cover.
    game_miss = game_cov.copy(); game_miss["home_score"] = 20; game_miss["away_score"] = 25
    out2 = score_pick(pred, game_miss, week=5)
    assert out2["covered"] is False, "home +3 losing by 5 must not cover"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
