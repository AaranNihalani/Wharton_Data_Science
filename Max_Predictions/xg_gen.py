# Packages
import csv
from load import load_games


# Main
if __name__ == "__main__":
    # Load games
    data = load_games("whl_2025.csv")
    print(f"Loaded {len(data)} games. \n\n")

    # Obtain xG/TOI*60 (expected goals/min) statistics
    csv_file = open('xg_toi.csv', 'w', newline='', encoding='utf-8')
    xg_toi_arr = []
    for game in data:
        for record in game.records:
            if record.toi != 0:
                away_xg_toi = record.away_xg / record.toi * 60
                home_xg_toi = record.home_xg / record.toi * 60
                for i in range(round(record.toi / 60)):
                    xg_toi_arr.append([away_xg_toi])
                    xg_toi_arr.append([home_xg_toi])
    results = csv.writer(csv_file)
    results.writerows(xg_toi_arr)
    print("Done! ")
