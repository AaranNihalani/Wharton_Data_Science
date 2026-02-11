# Packages
import csv
from math import exp, pi, sqrt
import pandas as pd

def g(phi_j: float) -> float:
    """Opponent scaling function in Glicko-2 rating system"""
    return pow(1 + 3.0 * pow(phi_j, 2) / pow(pi, 2), -0.5)

if __name__ == "__main__":
    # Load elos
    df = pd.read_csv("power_ranking.csv")

    # Evaluating elos
    elos = {}
    for _, row in df.iterrows():
        elos[row["country"]] = [row["rating"], row["rd (rating deviation)"]]

    # Load matchups
    match_df = pd.read_excel("WHSDSC_Rnd1_matchups.xlsx")
    predictions = []
    for _, row in match_df.iterrows():
        predictions.append({"game_id": row["game_id"],
                            "home_team": row["home_team"],
                            "away_team": row["away_team"],
                            "home_win_probability": pow(1 + exp(-g(sqrt(elos[row["away_team"]][1]))
                                                                * (elos[row["home_team"]][0] - elos[row["away_team"]][0])), -1) * 100})

    # Writing results to csv file
    with open("matchup_predictions.csv", 'w', newline='') as f:
        fieldnames = ["game_id", "home_team", "away_team", "home_win_probability"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    print("Done! ")
