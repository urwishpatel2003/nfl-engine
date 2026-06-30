import pandas as pd
from pathlib import Path

rw = pd.read_parquet(Path('data/raw/rosters_weekly.parquet'))
r26 = rw[rw['season'] == 2026]
print(f'2026 rows: {len(r26):,}  |  Teams: {r26["team"].nunique()}/32')
print()

def find(df, name):
    mask = pd.Series(True, index=df.index)
    for word in name.lower().split():
        mask &= df['player_name'].str.lower().str.contains(word, na=False)
    return df[mask]

checks = [
    ('Kyler Murray',     'MIN', 'Kyler Murray → MIN'),
    ('Tua Tagovailoa',   'ATL', 'Tua → ATL'),
    ('Kenneth Walker',   'KC',  'Kenneth Walker → KC'),
    ('DJ Moore',         'BUF', 'DJ Moore → BUF'),
    ('Mike Evans',       'SF',  'Mike Evans → SF'),
    ('Tyler Linderbaum', 'LV',  'Linderbaum → LV'),
    ('Nick Folk',        'ATL', 'Nick Folk → ATL'),
    ('Trey Hendrickson', 'BAL', 'Hendrickson → BAL'),
    ('Derrick Henry',    'BAL', 'Henry still BAL'),
    ('Jalen Hurts',      'PHI', 'Hurts still PHI'),
    ('Josh Allen',       'BUF', 'Josh Allen still BUF'),
    ('Drake Maye',       'NE',  'Drake Maye still NE'),
    ('Lamar Jackson',    'BAL', 'Lamar still BAL'),
    ('Patrick Mahomes',  'KC',  'Mahomes still KC'),
    ('Travis Etienne',   'NO',  'Etienne → NO'),
    ('David Montgomery', 'HOU', 'Montgomery → HOU'),
    ('Bradley Chubb',    'BUF', 'Chubb → BUF'),
    ('Rashan Gary',      'DAL', 'Rashan Gary → DAL'),
    ('Braden Smith',     'HOU', 'Braden Smith → HOU'),
    ('Isiah Pacheco',    'DET', 'Pacheco → DET'),
]

print(f'  {"Move":<34} Status')
print(f'  {"-"*50}')
ok = wrong = missing = 0
for name, team, label in checks:
    m = find(r26, name)
    if m.empty:
        print(f'  {label:<34} NOT FOUND')
        missing += 1
    else:
        actual = m.iloc[0]['team']
        if actual == team:
            print(f'  {label:<34} OK')
            ok += 1
        else:
            all_hits = ', '.join(f"{row.player_name}({row.team})" for _, row in m.iterrows())
            print(f'  {label:<34} WRONG -> {all_hits}')
            wrong += 1

print(f'\n  {ok} OK  |  {wrong} wrong  |  {missing} not found  ({len(checks)} total)')
print()
print('POSITION BREAKDOWN:')
for p, c in r26['position'].value_counts().head(12).items():
    print(f'  {str(p):<6} {c}')
print()
print('TEAM SIZES:')
sizes = r26.groupby('team').size().sort_values(ascending=False)
for i, (t, c) in enumerate(sizes.items()):
    print(f'  {t:<4} {c}', end='   ')
    if (i + 1) % 6 == 0:
        print()
print()
