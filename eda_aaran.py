import pandas as pd
import numpy as np

# Load the dataset
file_path = 'assets/whl_2025.csv'
df = pd.read_csv(file_path)

# 1. Team Performance Analysis
def get_team_stats(df):
    home_stats = df.groupby('home_team').agg({
        'home_goals': 'sum',
        'home_xg': 'sum',
        'home_shots': 'sum',
        'home_assists': 'sum',
        'home_penalty_minutes': 'sum',
        'game_id': 'nunique'
    }).rename(columns={
        'home_goals': 'goals_for',
        'home_xg': 'xg_for',
        'home_shots': 'shots_for',
        'home_assists': 'assists',
        'home_penalty_minutes': 'pim',
        'game_id': 'games'
    })

    away_stats = df.groupby('away_team').agg({
        'away_goals': 'sum',
        'away_xg': 'sum',
        'away_shots': 'sum',
        'away_assists': 'sum',
        'away_penalty_minutes': 'sum',
        'game_id': 'nunique'
    }).rename(columns={
        'away_goals': 'goals_for',
        'away_xg': 'xg_for',
        'away_shots': 'shots_for',
        'away_assists': 'assists',
        'away_penalty_minutes': 'pim',
        'game_id': 'games'
    })

    team_stats = home_stats.add(away_stats, fill_value=0)
    
    home_against = df.groupby('home_team').agg({
        'away_goals': 'sum',
        'away_xg': 'sum'
    }).rename(columns={'away_goals': 'goals_against', 'away_xg': 'xg_against'})
    
    away_against = df.groupby('away_team').agg({
        'home_goals': 'sum',
        'home_xg': 'sum'
    }).rename(columns={'home_goals': 'goals_against', 'home_xg': 'xg_against'})
    
    team_against = home_against.add(away_against, fill_value=0)
    team_full_stats = pd.concat([team_stats, team_against], axis=1)
    
    team_full_stats['goal_diff'] = team_full_stats['goals_for'] - team_full_stats['goals_against']
    team_full_stats['xg_diff'] = team_full_stats['xg_for'] - team_full_stats['xg_against']
    team_full_stats['goals_per_game'] = team_full_stats['goals_for'] / team_full_stats['games']
    team_full_stats['shooting_pct'] = (team_full_stats['goals_for'] / team_full_stats['shots_for']) * 100
    
    return team_full_stats

team_performance = get_team_stats(df)

print("--- Top 5 Teams by Goal Difference ---")
print(team_performance.sort_values(by='goal_diff', ascending=False)[['goals_for', 'goals_against', 'goal_diff', 'games']].head())

print("\n--- Top 5 Teams by Expected Goal Difference (xG Diff) ---")
print(team_performance.sort_values(by='xg_diff', ascending=False)[['xg_for', 'xg_against', 'xg_diff']].head())

# 2. Goalie Analysis
def get_goalie_stats(df):
    # Home goalies (faced away shots/xg)
    home_goalie_stats = df.groupby('home_goalie').agg({
        'away_goals': 'sum',
        'away_xg': 'sum',
        'away_shots': 'sum',
        'toi': 'sum'
    }).rename(columns={
        'away_goals': 'goals_against',
        'away_xg': 'xg_against',
        'away_shots': 'shots_against',
        'toi': 'toi'
    })
    
    # Away goalies (faced home shots/xg)
    away_goalie_stats = df.groupby('away_goalie').agg({
        'home_goals': 'sum',
        'home_xg': 'sum',
        'home_shots': 'sum',
        'toi': 'sum'
    }).rename(columns={
        'home_goals': 'goals_against',
        'home_xg': 'xg_against',
        'home_shots': 'shots_against',
        'toi': 'toi'
    })
    
    goalie_stats = home_goalie_stats.add(away_goalie_stats, fill_value=0)
    goalie_stats = goalie_stats[goalie_stats.index != 'empty_net']
    
    goalie_stats['save_pct'] = (1 - (goalie_stats['goals_against'] / goalie_stats['shots_against'])) * 100
    goalie_stats['goals_saved_above_expected'] = goalie_stats['xg_against'] - goalie_stats['goals_against']
    
    return goalie_stats

goalie_performance = get_goalie_stats(df)
print("\n--- Top 5 Goalies by Goals Saved Above Expected (GSAx) ---")
print(goalie_performance.sort_values(by='goals_saved_above_expected', ascending=False)[['goals_against', 'xg_against', 'goals_saved_above_expected']].head())

# 3. Game-level analysis
game_totals = df.groupby('game_id').agg({
    'home_goals': 'sum',
    'away_goals': 'sum',
    'went_ot': 'max'
})
game_totals['total_goals'] = game_totals['home_goals'] + game_totals['away_goals']

print("\n--- Game Statistics ---")
print(f"Average goals per game: {game_totals['total_goals'].mean():.2f}")
print(f"Percentage of games going to OT: {(game_totals['went_ot'].mean() * 100):.2f}%")

# 4. Finishing Efficiency
team_performance['finishing_efficiency'] = team_performance['goals_for'] - team_performance['xg_for']
print("\n--- Best Finishing Teams (Goals - xG) ---")
print(team_performance.sort_values(by='finishing_efficiency', ascending=False)[['goals_for', 'xg_for', 'finishing_efficiency']].head())
