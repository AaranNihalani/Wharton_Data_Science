import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('../assets/whl_2025.csv')

# --- 1. Calculate Defensive Strength of Every Pairing ---
# We want to know how many xG per 60 each defensive pairing allows.
# A higher value means a weaker defense.

# Prepare Home Defense Data (facing Away Offense)
home_def = df[['home_team', 'home_def_pairing', 'away_xg', 'toi']].copy()
home_def.columns = ['team', 'pairing', 'xg_allowed', 'toi']

# Prepare Away Defense Data (facing Home Offense)
away_def = df[['away_team', 'away_def_pairing', 'home_xg', 'toi']].copy()
away_def.columns = ['team', 'pairing', 'xg_allowed', 'toi']

# Combine
all_def = pd.concat([home_def, away_def])

# Group by Team and Pairing
def_stats = all_def.groupby(['team', 'pairing']).agg({
    'xg_allowed': 'sum',
    'toi': 'sum'
}).reset_index()

# Calculate xG Allowed per 60
def_stats['xg_allowed_per_60'] = (def_stats['xg_allowed'] / def_stats['toi']) * 3600

# Calculate League Average xG Allowed per 60 (Global Baseline)
league_avg_xg_per_60 = (all_def['xg_allowed'].sum() / all_def['toi'].sum()) * 3600
print(f"League Average xG Allowed per 60: {league_avg_xg_per_60:.4f}")

# Map (Team, Pairing) -> Rating
def_rating_map = def_stats.set_index(['team', 'pairing'])['xg_allowed_per_60'].to_dict()

# --- 2. Calculate Offensive Performance Adjusted for Defense ---
relevant_lines = ['first_off', 'second_off']

home_off = df[df['home_off_line'].isin(relevant_lines)][['home_team', 'home_off_line', 'away_team', 'away_def_pairing', 'home_xg', 'toi']].copy()
home_off.columns = ['team', 'line', 'opp_team', 'opp_pairing', 'xg_for', 'toi']

away_off = df[df['away_off_line'].isin(relevant_lines)][['away_team', 'away_off_line', 'home_team', 'home_def_pairing', 'away_xg', 'toi']].copy()
away_off.columns = ['team', 'line', 'opp_team', 'opp_pairing', 'xg_for', 'toi']

all_off = pd.concat([home_off, away_off])

def get_opp_rating(row):
    return def_rating_map.get((row['opp_team'], row['opp_pairing']), league_avg_xg_per_60)

all_off['opp_def_rating'] = all_off.apply(get_opp_rating, axis=1)

off_stats = all_off.groupby(['team', 'line']).apply(
    lambda x: pd.Series({
        'total_xg': x['xg_for'].sum(),
        'total_toi': x['toi'].sum(),
        'avg_opp_def_rating': np.average(x['opp_def_rating'], weights=x['toi'])
    }),
    include_groups=False
).reset_index()

off_stats['raw_xg_per_60'] = (off_stats['total_xg'] / off_stats['total_toi']) * 3600
off_stats['adj_xg_per_60'] = off_stats['raw_xg_per_60'] * (league_avg_xg_per_60 / off_stats['avg_opp_def_rating'])

# --- 3. Calculate Defensive Performance (for comparison) ---
# We focus on 'first_def' and 'second_def' pairings
relevant_pairings = ['first_def', 'second_def']
def_stats_filtered = def_stats[def_stats['pairing'].isin(relevant_pairings)].copy()

# --- 4. Prepare Final Dataset for Visualization ---
# Pivot Offense
off_pivot = off_stats.pivot(index='team', columns='line', values='adj_xg_per_60').reset_index()
off_pivot['off_disparity'] = off_pivot['first_off'] / off_pivot['second_off']

# Pivot Defense (Note: Lower is better for defense, so we might want to invert or just show raw)
def_pivot = def_stats_filtered.pivot(index='team', columns='pairing', values='xg_allowed_per_60').reset_index()
# For defense, if first_def allows less than second_def, that's "better".
# Ratio > 1 means first pairing is WORSE (allows more goals) than second pairing.
# Ratio < 1 means first pairing is BETTER (allows fewer goals).
# To make it comparable to offense (where > 1 means 1st line is "more productive"), 
# let's calculate: Second / First.
# If Second allows 3.0 and First allows 2.0, Ratio = 1.5. (First is 1.5x better/stingier).
def_pivot['def_disparity'] = def_pivot['second_def'] / def_pivot['first_def']

# Merge everything
final_df = pd.merge(off_pivot, def_pivot, on='team')
final_df.to_csv('line_analysis_full.csv', index=False)
print("Saved full line analysis to line_analysis_full.csv")
