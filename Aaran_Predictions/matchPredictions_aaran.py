import pandas as pd
import numpy as np

# Load power rankings data
rankings_df = pd.read_csv('power_rankings_data.csv')
team_strengths = dict(zip(rankings_df['team'], rankings_df['strength_score']))

# Load matchups
matchups_df = pd.read_excel('assets/WHSDSC_Rnd1_matchups.xlsx')

def predict_win_probability(home_team, away_team, strengths):
    beta_h = strengths.get(home_team, 0)
    beta_a = strengths.get(away_team, 0)
    
    # Bradley-Terry probability formula
    prob = 1 / (1 + np.exp(-(beta_h - beta_a)))
    return prob

# Calculate win probabilities
matchups_df['home_win_probability'] = matchups_df.apply(
    lambda row: predict_win_probability(row['home_team'], row['away_team'], team_strengths), axis=1
)

print("--- Rnd 1 Matchup Predictions ---")
print(matchups_df[['game_id', 'home_team', 'away_team', 'home_win_probability']].to_string(index=False))

# Save predictions to CSV
matchups_df.to_csv('matchup_predictions_aaran.csv', index=False)
