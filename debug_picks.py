"""
debug_picks.py — Diagnose why backtest_picks returns 0/0
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

RAW  = Path(__file__).parent / "data" / "raw"
PROC = Path(__file__).parent / "data" / "processed"

from engine.predict import load_engine_data, predict_game
from engine.styles import build_team_styles

print("="*60)
print("STEP 1: Load base engine data")
data = load_engine_data()
print(f"  composite rows:  {len(data.get('composite', pd.DataFrame())):,}")
print(f"  styles rows:     {len(data.get('styles', pd.DataFrame())):,}")
print(f"  schedules rows:  {len(data.get('schedules', pd.DataFrame())):,}")
cond = data.get('conditions')
cond_rows = len(cond) if cond is not None and not isinstance(cond, type(None)) else 0
print(f"  conditions rows: {cond_rows:,}")

print()
print("STEP 2: Check schedules spread lines for 2025")
sched = pd.read_parquet(RAW / "schedules.parquet")
sched["season"]    = pd.to_numeric(sched["season"], errors="coerce").astype("Int64")
sched["game_type"] = sched["game_type"].str.upper().str.strip()
s25 = sched[(sched["season"]==2025) & (sched["game_type"]=="REG")]
print(f"  2025 REG games: {len(s25)}")
print(f"  With spread_line: {s25['spread_line'].notna().sum()}")
print(f"  spread_line sample: {s25['spread_line'].dropna().head(5).tolist()}")

print()
print("STEP 3: Run ONE prediction (2025 Wk1)")
wk1 = s25[s25["week"]==1].head(3)
for _, game in wk1.iterrows():
    home = game["home_team"]
    away = game["away_team"]
    spread = game["spread_line"]
    try:
        pred = predict_game(home, away, 2025, 1, data=data)
        home_p = pred.get("predicted_home_score", 0)
        away_p = pred.get("predicted_away_score", 0)
        pred_margin  = home_p - away_p
        vegas_margin = float(spread)  # positive spread_line = home favored (home margin)
        model_home   = pred_margin > 0
        vegas_home   = spread > 0
        agrees = model_home == vegas_home
        print(f"  {away}@{home}")
        print(f"    Vegas spread: {spread:+.1f}  (home margin: {vegas_margin:+.1f})")
        print(f"    Model pred:   {away_p:.1f}-{home_p:.1f}  (home margin: {pred_margin:+.1f})")
        print(f"    Model home:   {model_home}  Vegas home: {vegas_home}  Agrees: {agrees}")
        print(f"    Gap: {abs(pred_margin - vegas_margin):.1f} pts")
        print()
    except Exception as e:
        print(f"  {away}@{home}: ERROR — {e}")

print()
print("STEP 4: Check how many games model DISAGREES with Vegas (2025 wk1-3)")
wk3 = s25[s25["week"]<=3]
disagree = 0
total = 0
for _, game in wk3.iterrows():
    try:
        pred = predict_game(str(game["home_team"]), str(game["away_team"]),
                            2025, int(game["week"]), data=data)
        home_p = pred.get("predicted_home_score", 22)
        away_p = pred.get("predicted_away_score", 22)
        spread = float(game["spread_line"])
        pred_margin  = home_p - away_p
        model_home   = pred_margin > 0
        vegas_home   = spread > 0
        total += 1
        if model_home != vegas_home:
            disagree += 1
    except:
        pass

print(f"  Games wk1-3: {total}  Model disagrees: {disagree} ({disagree/total*100:.0f}% if total>0)")

print()
print("STEP 5: Test with OOS styles (2021-2022 only)")
styles_oos = build_team_styles([2021, 2022])
data_oos = dict(data)
data_oos["styles"] = styles_oos

disagree_oos = 0
for _, game in wk3.iterrows():
    try:
        pred = predict_game(str(game["home_team"]), str(game["away_team"]),
                            2023, int(game["week"]), data=data_oos)
        home_p = pred.get("predicted_home_score", 22)
        away_p = pred.get("predicted_away_score", 22)
        spread = float(game["spread_line"])
        pred_margin = home_p - away_p
        if (pred_margin > 0) != (spread > 0):
            disagree_oos += 1
    except Exception as e:
        pass

print(f"  With OOS styles — disagrees: {disagree_oos}/{total}")

print()
print("STEP 6: Check data['schedules'] vs raw schedules")
data_sched = data.get("schedules", pd.DataFrame())
print(f"  data['schedules'] rows: {len(data_sched)}")
if not data_sched.empty:
    print(f"  seasons in data schedules: {sorted(data_sched['season'].dropna().unique().tolist())}")
    s25_data = data_sched[(data_sched["season"]==2025)]
    print(f"  2025 rows in data schedules: {len(s25_data)}")
    print(f"  spread_line notna: {s25_data['spread_line'].notna().sum() if 'spread_line' in s25_data.columns else 'NO SPREAD COL'}")
