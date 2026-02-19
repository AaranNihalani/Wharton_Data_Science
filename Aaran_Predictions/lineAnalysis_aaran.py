import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('assets/whl_2025.csv')

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
# We create a dictionary for fast lookup: (team, pairing) -> xG_allowed_per_60
def_rating_map = def_stats.set_index(['team', 'pairing'])['xg_allowed_per_60'].to_dict()

# --- 2. Calculate Offensive Performance Adjusted for Defense ---
# We focus only on 'first_off' and 'second_off' lines.

# Filter for relevant offensive lines
relevant_lines = ['first_off', 'second_off']

# Prepare Home Offense Data (facing Away Defense)
home_off = df[df['home_off_line'].isin(relevant_lines)][['home_team', 'home_off_line', 'away_team', 'away_def_pairing', 'home_xg', 'toi']].copy()
home_off.columns = ['team', 'line', 'opp_team', 'opp_pairing', 'xg_for', 'toi']

# Prepare Away Offense Data (facing Home Defense)
away_off = df[df['away_off_line'].isin(relevant_lines)][['away_team', 'away_off_line', 'home_team', 'home_def_pairing', 'away_xg', 'toi']].copy()
away_off.columns = ['team', 'line', 'opp_team', 'opp_pairing', 'xg_for', 'toi']

# Combine
all_off = pd.concat([home_off, away_off])

# Add Opponent Defensive Rating to each row
def get_opp_rating(row):
    return def_rating_map.get((row['opp_team'], row['opp_pairing']), league_avg_xg_per_60)

all_off['opp_def_rating'] = all_off.apply(get_opp_rating, axis=1)

# Group by Team and Line
# Using include_groups=False to avoid deprecation warning
off_stats = all_off.groupby(['team', 'line']).apply(
    lambda x: pd.Series({
        'total_xg': x['xg_for'].sum(),
        'total_toi': x['toi'].sum(),
        'avg_opp_def_rating': np.average(x['opp_def_rating'], weights=x['toi'])
    }),
    include_groups=False
).reset_index()

# Calculate Raw xG per 60
off_stats['raw_xg_per_60'] = (off_stats['total_xg'] / off_stats['total_toi']) * 3600

# Calculate Adjusted xG per 60
# Logic: If you played against defenses that allow 2.0 (avg 1.5), your stats are inflated.
# Adj = Raw * (League_Avg / Opp_Avg)
off_stats['adj_xg_per_60'] = off_stats['raw_xg_per_60'] * (league_avg_xg_per_60 / off_stats['avg_opp_def_rating'])

# --- 3. Calculate Disparity Ratio (First / Second) ---
# Pivot to get columns for first and second
pivoted = off_stats.pivot(index='team', columns='line', values='adj_xg_per_60').reset_index()

if 'first_off' not in pivoted.columns or 'second_off' not in pivoted.columns:
    print("Error: Missing first or second line data for some teams.")
else:
    # Calculate Ratio: First / Second
    pivoted['disparity_ratio'] = pivoted['first_off'] / pivoted['second_off']
    
    # Sort by ratio descending
    ranked_teams = pivoted.sort_values(by='disparity_ratio', ascending=False).reset_index(drop=True)
    ranked_teams['rank'] = ranked_teams.index + 1
    
    # Display Top 10
    print("\n--- Top 10 Teams by Offensive Line Quality Disparity (1st vs 2nd) ---")
    print(ranked_teams[['rank', 'team', 'disparity_ratio', 'first_off', 'second_off']].head(10).to_string(index=False))
    
    # Save to CSV
    ranked_teams.to_csv('line_disparity_analysis.csv', index=False)
