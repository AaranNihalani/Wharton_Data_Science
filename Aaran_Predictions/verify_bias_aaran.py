import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the dataset
file_path = '../Assets/whl_2025.csv'
df = pd.read_csv(file_path)

# Aggregate data by game
game_results = df.groupby(['game_id', 'home_team', 'away_team']).agg({
    'home_goals': 'sum',
    'away_goals': 'sum'
}).reset_index()
game_results['home_win'] = (game_results['home_goals'] > game_results['away_goals']).astype(int)

teams = sorted(list(set(game_results['home_team']).union(set(game_results['away_team']))))
team_to_idx = {team: i for i, team in enumerate(teams)}

X = np.zeros((len(game_results), len(teams)))
for i, row in game_results.iterrows():
    X[i, team_to_idx[row['home_team']]] = 1
    X[i, team_to_idx[row['away_team']]] = -1

y = game_results['home_win']

def run_bt_model(reference_idx):
    # Remove the reference team column
    cols = [i for i in range(len(teams)) if i != reference_idx]
    X_reduced = X[:, cols]
    
    model = sm.Logit(y, X_reduced).fit(disp=0)
    
    # Reconstruct coefficients including the reference team (set to 0)
    coeffs = np.zeros(len(teams))
    coeffs[cols] = model.params
    return coeffs

# Run with Brazil as reference (index 0)
coeffs_ref0 = run_bt_model(0)
# Run with Vietnam as reference (index 31 - last team)
coeffs_ref31 = run_bt_model(31)

# Create comparison DataFrame
comparison = pd.DataFrame({
    'team': teams,
    'score_ref_brazil': coeffs_ref0,
    'score_ref_vietnam': coeffs_ref31
})

# Normalize both so mean is 0 for direct comparison
comparison['norm_brazil'] = comparison['score_ref_brazil'] - comparison['score_ref_brazil'].mean()
comparison['norm_vietnam'] = comparison['score_ref_vietnam'] - comparison['score_ref_vietnam'].mean()

# Calculate ranks
comparison['rank_brazil'] = comparison['norm_brazil'].rank(ascending=False).astype(int)
comparison['rank_vietnam'] = comparison['norm_vietnam'].rank(ascending=False).astype(int)

# Check for differences
rank_diffs = (comparison['rank_brazil'] != comparison['rank_vietnam']).sum()
score_corr = comparison['norm_brazil'].corr(comparison['norm_vietnam'])

print("--- Reference Team Bias Verification ---")
print(f"Number of rank mismatches: {rank_diffs}")
print(f"Correlation between scores: {score_corr:.10f}")

print("\n--- Sample Comparison (Top 5 by Brazil Ref) ---")
print(comparison.sort_values(by='rank_brazil')[['team', 'rank_brazil', 'rank_vietnam', 'norm_brazil', 'norm_vietnam']].head())

if rank_diffs == 0 and score_corr > 0.99999:
    print("\nSUCCESS: The choice of reference team does NOT affect the rankings or relative strengths.")
else:
    print("\nWARNING: There are differences between the models.")
