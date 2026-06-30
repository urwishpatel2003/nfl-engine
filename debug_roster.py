import pandas as pd
from pathlib import Path

rw = pd.read_parquet(Path('data/raw/rosters_weekly.parquet'))
rows_2026 = rw[rw['season'] == 2026]

print("NAME COLLISION DEBUG")
print("="*55)

collisions = [
    ('Walker',      'KC',  'Kenneth Walker'),
    ('Moore',       'BUF', 'DJ Moore'),
    ('Evans',       'SF',  'Mike Evans'),
    ('Henry',       'BAL', 'Derrick Henry'),
    ('Allen',       'BUF', 'Josh Allen'),
]

for last, expected_team, full_name in collisions:
    matches = rows_2026[rows_2026['player_name'].str.contains(last, na=False, case=False)]
    print(f"\n'{last}' matches ({len(matches)} players):")
    for _, r in matches[['player_name','team','position']].iterrows():
        marker = " ← WANT" if r['team'] == expected_team else ""
        print(f"  {r['player_name']:<28} {r['team']:<5} {r['position']}{marker}")

print()
print("="*55)
print("FULL NAME CHECKS:")
full_checks = [
    ('Kenneth Walker',  'KC'),
    ('DJ Moore',        'BUF'),
    ('DeVante',         'BUF'),  # DJ Moore's legal name
    ('Mike Evans',      'SF'),
    ('Derrick Henry',   'BAL'),
    ('Josh Allen',      'BUF'),
]
for name, team in full_checks:
    parts = name.lower().split()
    mask = rows_2026['player_name'].str.lower().str.contains(parts[0], na=False)
    for p in parts[1:]:
        mask &= rows_2026['player_name'].str.lower().str.contains(p, na=False)
    m = rows_2026[mask]
    print(f"\n'{name}':")
    for _, r in m[['player_name','team','position']].iterrows():
        marker = " ← WANT" if r['team'] == team else ""
        print(f"  {r['player_name']:<28} {r['team']:<5} {r['position']}{marker}")
