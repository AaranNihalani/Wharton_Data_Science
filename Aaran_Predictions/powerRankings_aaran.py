import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the dataset
file_path = '../Assets/whl_2025.csv'
df = pd.read_csv(file_path)

# Aggregate data by game to get total goals and determine winner
game_results = df.groupby(['game_id', 'home_team', 'away_team']).agg({
    'home_goals': 'sum',
    'away_goals': 'sum'
}).reset_index()

# Determine winner (1 for home win, 0 for away win)
# Note: In case of a tie, we'll treat it as 0.5 or drop it? 
# Bradley-Terry usually assumes a winner. 
# Let's check for ties.
game_results['home_win'] = (game_results['home_goals'] > game_results['away_goals']).astype(int)
ties = game_results[game_results['home_goals'] == game_results['away_goals']]
if len(ties) > 0:
    # If there are ties, we can exclude them or handle them. 
    # For simplicity in Bradley-Terry, we often drop ties or use a different model.
    # However, hockey games usually have a winner (OT/SO). 
    # The 'went_ot' column exists, so let's check if there are any actual ties in goals.
    pass

# Prepare data for Bradley-Terry Logistic Regression
teams = sorted(list(set(game_results['home_team']).union(set(game_results['away_team']))))
team_to_idx = {team: i for i, team in enumerate(teams)}

X = np.zeros((len(game_results), len(teams)))
for i, row in game_results.iterrows():
    X[i, team_to_idx[row['home_team']]] = 1
    X[i, team_to_idx[row['away_team']]] = -1

# Drop one team to avoid perfect multicollinearity (reference team)
# We'll use the first team as reference
X_reduced = X[:, 1:]
y = game_results['home_win']

# Fit logistic regression
# We don't use an intercept because the team strengths are relative
model = sm.Logit(y, X_reduced).fit()

# Get coefficients and add 0 for the reference team
coefficients = np.insert(model.params, 0, 0)

# Create Power Rankings
power_rankings = pd.DataFrame({
    'team': teams,
    'strength_score': coefficients
})

# Normalize scores (Mean-Zero Normalization)
# This makes the scores independent of the reference team choice
power_rankings['strength_score'] = power_rankings['strength_score'] - power_rankings['strength_score'].mean()

power_rankings = power_rankings.sort_values(by='strength_score', ascending=False).reset_index(drop=True)
power_rankings['rank'] = power_rankings.index + 1

print("--- Bradley-Terry Power Rankings ---")
print(power_rankings[['rank', 'team', 'strength_score']].to_string(index=False))

# Save to CSV for the prediction script to use
power_rankings.to_csv('power_rankings_data.csv', index=False)
