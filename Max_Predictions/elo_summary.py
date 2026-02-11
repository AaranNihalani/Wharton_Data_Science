# Packages
import csv
from dataclasses import dataclass
from glicko2 import Player
from load import Game, load_games
from math import exp, pi, pow, sqrt
import scipy.stats as stats
from typing import List, Dict

# Constants
ALPHA = 0.2097803028996198
BETA = 0.12146177505214789
LOC = 0
STDDEV = sqrt(stats.gamma.var(ALPHA, loc=LOC, scale=BETA))


@dataclass
class Elo:
    """All elo values of one ice hockey team"""
    def_pairs_elo: Dict[str, Player]
    off_lines_elo: Dict[str, Player]
    penalty_lines_elo: Dict[str, Player]
    goalie_elo: Player

def create_elo(
    def_pairs_elo: Dict[str, Player] = None,
    off_lines_elo: Dict[str, Player] = None,
    penalty_lines_elo: Dict[str, Player] = None,
    goalie_elo: float = None
):
    """Elo dataclass constructor"""
    if def_pairs_elo is None:
        def_pairs_elo = {"first_def": Player(), "second_def": Player(), "empty_net_line": Player()}
    if off_lines_elo is None:
        off_lines_elo = {"first_off": Player(), "second_off": Player(), "empty_net_line": Player()}
    if penalty_lines_elo is None:
        penalty_lines_elo = {"PP_up": Player(), "PP_kill_dwn": Player()}
    if goalie_elo is None:
        goalie_elo = Player()

    return Elo(def_pairs_elo=def_pairs_elo,
               off_lines_elo=off_lines_elo,
               penalty_lines_elo=penalty_lines_elo,
               goalie_elo=goalie_elo)

def g(phi_j: float) -> float:
    """Opponent scaling function in Glicko-2 rating system"""
    return pow(1 + 3.0 * pow(phi_j, 2) / pow(pi, 2), -0.5)

def generate_elo(games: List[Game]) -> Dict[str, Elo]:
    """Generate elos for each team based upon every game"""
    elo_list: Dict[str, Elo] = {}
    for game in games:
        for record in game.records:
            try:
                elo_list[record.home_team]
            except KeyError:
                elo_list[record.home_team] = create_elo()
            try:
                elo_list[record.away_team]
            except KeyError:
                elo_list[record.away_team] = create_elo()

            # Shots elo change
            if record.away_off_line in ["PP_up", "PP_kill_dwn"]:
                away_off_line_rating = elo_list[record.away_team].penalty_lines_elo[record.away_off_line].getRating()
                away_off_line_rd = elo_list[record.away_team].penalty_lines_elo[record.away_off_line].getRd()
            else:
                away_off_line_rating = elo_list[record.away_team].off_lines_elo[record.away_off_line].getRating()
                away_off_line_rd = elo_list[record.away_team].off_lines_elo[record.away_off_line].getRd()
            if record.home_off_line in ["PP_up", "PP_kill_dwn"]:
                home_off_line_rating = elo_list[record.home_team].penalty_lines_elo[record.home_off_line].getRating()
                home_off_line_rd = elo_list[record.home_team].penalty_lines_elo[record.home_off_line].getRd()
            else:
                home_off_line_rating = elo_list[record.home_team].off_lines_elo[record.home_off_line].getRating()
                home_off_line_rd = elo_list[record.home_team].off_lines_elo[record.home_off_line].getRd()
            
            # Away team shots
            if record.away_shots != 0 and record.home_goalie != "empty_net":
                # Elo parameters
                goalie_rating = elo_list[record.home_team].goalie_elo.getRating()
                goalie_rd = elo_list[record.home_team].goalie_elo.getRd()

                # All shots
                elo_list[record.home_team].goalie_elo.update_player(
                    [away_off_line_rating for _ in range(record.away_shots)],
                    [away_off_line_rd for _ in range(record.away_shots)],
                    [False for _ in range(record.away_goals)]
                    + [True for _ in range(record.away_shots - record.away_goals)])
                if record.away_off_line in ["PP_up", "PP_kill_dwn"]:
                    elo_list[record.away_team].penalty_lines_elo[record.away_off_line].update_player(
                        [goalie_rating for _ in range(record.away_shots)],
                        [goalie_rd for _ in range(record.away_shots)],
                        [True for _ in range(record.away_goals)] 
                        + [False for _ in range(record.away_shots - record.away_goals)])
                else:
                    elo_list[record.away_team].off_lines_elo[record.away_off_line].update_player(
                        [goalie_rating for _ in range(record.away_shots)],
                        [goalie_rd for _ in range(record.away_shots)],
                        [True for _ in range(record.away_goals)]
                        + [False for _ in range(record.away_shots - record.away_goals)])

            # Home team shots
            if record.home_shots != 0 and record.away_goalie != "empty_net":
                # Elo parameters
                goalie_rating = elo_list[record.away_team].goalie_elo.getRating()
                goalie_rd = elo_list[record.away_team].goalie_elo.getRd()

                if record.home_off_line in ["PP_up", "PP_kill_dwn"]:
                    home_off_line_rd = elo_list[record.home_team].penalty_lines_elo[record.home_off_line].getRd()
                else:
                    home_off_line_rd = elo_list[record.home_team].off_lines_elo[record.home_off_line].getRd()

                # Expected number of goals
                elo_list[record.away_team].goalie_elo.update_player(
                    [home_off_line_rating for _ in range(record.home_shots)],
                    [home_off_line_rd for _ in range(record.home_shots)],
                    [False for _ in range(record.home_goals)]
                    + [True for _ in range(record.home_shots - record.home_goals)])
                if record.home_off_line in ["PP_up", "PP_kill_dwn"]:
                    elo_list[record.home_team].penalty_lines_elo[record.home_off_line].update_player(
                        [goalie_rating for _ in range(record.home_shots)],
                        [goalie_rd for _ in range(record.home_shots)],
                        [True for _ in range(record.home_goals)] 
                        + [False for _ in range(record.home_shots - record.home_goals)])
                else:
                    elo_list[record.home_team].off_lines_elo[record.home_off_line].update_player(
                        [goalie_rating for _ in range(record.home_shots)],
                        [goalie_rd for _ in range(record.home_shots)],
                        [True for _ in range(record.home_goals)]
                        + [False for _ in range(record.home_shots - record.home_goals)])

            # Offensive defensive elos
            if record.away_def_pairing in ["PP_up", "PP_kill_dwn"]:
                away_def_pair_rating = elo_list[record.away_team].penalty_lines_elo[record.away_def_pairing].getRating()
                away_def_pair_rd = elo_list[record.away_team].penalty_lines_elo[record.away_def_pairing].getRd()
            else:
                away_def_pair_rating = elo_list[record.away_team].def_pairs_elo[record.away_def_pairing].getRating()
                away_def_pair_rd = elo_list[record.away_team].def_pairs_elo[record.away_def_pairing].getRd()
            if record.home_def_pairing in ["PP_up", "PP_kill_dwn"]:
                home_def_pair_rating = elo_list[record.home_team].penalty_lines_elo[record.home_def_pairing].getRating()
                home_def_pair_rd = elo_list[record.home_team].penalty_lines_elo[record.home_def_pairing].getRd()
            else:
                home_def_pair_rating = elo_list[record.home_team].def_pairs_elo[record.home_def_pairing].getRating()
                home_def_pair_rd = elo_list[record.home_team].def_pairs_elo[record.home_def_pairing].getRd()
            
            # Predicted xG/TOI*60 vs actual xG/TOI*60 and elo changes
            away_win = pow(1 + exp(-g(home_def_pair_rd) * (away_off_line_rating - home_def_pair_rating)), -1)
            home_win = pow(1 + exp(-g(away_def_pair_rd) * (home_off_line_rating - away_def_pair_rating)), -1)
            away_xg_toi_exp = stats.gamma.ppf(away_win, ALPHA, loc=0, scale=BETA)
            home_xg_toi_exp = stats.gamma.ppf(home_win, ALPHA, loc=0, scale=BETA)
            if record.toi == 0:
                continue
            away_xg_toi_act = record.away_xg / record.toi * 60
            home_xg_toi_act = record.home_xg / record.toi * 60
            mins = record.toi / 60
            if mins == 0:
                continue
            away_performance = abs(away_xg_toi_act - away_xg_toi_exp) / STDDEV
            home_performance = abs(home_xg_toi_act - home_xg_toi_exp) / STDDEV
            away_its = mins / 2
            home_its = mins / 2
            away_its *= 1 + away_performance / 15
            home_its *= 1 + home_performance / 15
            away_its = round(away_its)
            home_its = round(home_its)
            if away_its == 0:
                away_its = 1
            if home_its == 0:
                home_its = 1
            if away_xg_toi_act > away_xg_toi_exp and record.away_off_line != "PP_kill_dwn":
                if record.home_def_pairing in ["PP_up", "PP_kill_dwn"]:
                    elo_list[record.home_team].penalty_lines_elo[record.home_def_pairing].update_player(
                        [away_off_line_rating for _ in range(away_its)],
                        [away_off_line_rd for _ in range(away_its)],
                        [False for _ in range(away_its)])
                else:
                    elo_list[record.home_team].def_pairs_elo[record.home_def_pairing].update_player(
                        [away_off_line_rating for _ in range(away_its)],
                        [away_off_line_rd for _ in range(away_its)],
                        [False for _ in range(away_its)])
                if record.away_off_line in ["PP_up", "PP_kill_dwn"]:
                    elo_list[record.away_team].penalty_lines_elo[record.away_off_line].update_player(
                        [home_def_pair_rating for _ in range(away_its)],
                        [home_def_pair_rd for _ in range(away_its)],
                        [True for _ in range(away_its)])
                else:
                    elo_list[record.away_team].off_lines_elo[record.away_off_line].update_player(
                        [home_def_pair_rating for _ in range(away_its)],
                        [home_def_pair_rd for _ in range(away_its)],
                        [True for _ in range(away_its)])
            elif record.away_off_line != "PP_kill_dwn":
                if record.home_def_pairing in ["PP_up", "PP_kill_dwn"]:
                    elo_list[record.home_team].penalty_lines_elo[record.home_def_pairing].update_player(
                        [away_off_line_rating for _ in range(away_its)],
                        [away_off_line_rd for _ in range(away_its)],
                        [True for _ in range(away_its)])
                else:
                    elo_list[record.home_team].def_pairs_elo[record.home_def_pairing].update_player(
                        [away_off_line_rating for _ in range(away_its)],
                        [away_off_line_rd for _ in range(away_its)],
                        [True for _ in range(away_its)])
                if record.away_off_line in ["PP_up", "PP_kill_dwn"]:
                    elo_list[record.away_team].penalty_lines_elo[record.away_off_line].update_player(
                        [home_def_pair_rating for _ in range(away_its)],
                        [home_def_pair_rd for _ in range(away_its)],
                        [False for _ in range(away_its)])
                else:
                    elo_list[record.away_team].off_lines_elo[record.away_off_line].update_player(
                        [home_def_pair_rating for _ in range(away_its)],
                        [home_def_pair_rd for _ in range(away_its)],
                        [False for _ in range(away_its)])
            if home_xg_toi_act > home_xg_toi_exp and record.home_off_line != "PP_kill_dwn":
                if record.away_def_pairing in ["PP_up", "PP_kill_dwn"]:
                    elo_list[record.away_team].penalty_lines_elo[record.away_def_pairing].update_player(
                        [home_off_line_rating for _ in range(home_its)],
                        [home_off_line_rd for _ in range(home_its)],
                        [False for _ in range(home_its)])
                else:
                    elo_list[record.away_team].def_pairs_elo[record.away_def_pairing].update_player(
                        [home_off_line_rating for _ in range(home_its)],
                        [home_off_line_rd for _ in range(home_its)],
                        [False for _ in range(home_its)])
                if record.home_off_line in ["PP_up", "PP_kill_dwn"]:
                    elo_list[record.home_team].penalty_lines_elo[record.home_off_line].update_player(
                        [away_def_pair_rating for _ in range(home_its)],
                        [away_def_pair_rd for _ in range(home_its)],
                        [True for _ in range(home_its)])
                else:
                    elo_list[record.home_team].off_lines_elo[record.home_off_line].update_player(
                        [away_def_pair_rating for _ in range(home_its)],
                        [away_def_pair_rd for _ in range(home_its)],
                        [True for _ in range(home_its)])
            elif record.home_off_line != "PP_kill_dwn":
                if record.away_def_pairing in ["PP_up", "PP_kill_dwn"]:
                    elo_list[record.away_team].penalty_lines_elo[record.away_def_pairing].update_player(
                        [home_off_line_rating for _ in range(home_its)],
                        [home_off_line_rd for _ in range(home_its)],
                        [True for _ in range(home_its)])
                else:
                    elo_list[record.away_team].def_pairs_elo[record.away_def_pairing].update_player(
                        [home_off_line_rating for _ in range(home_its)],
                        [home_off_line_rd for _ in range(home_its)],
                        [True for _ in range(home_its)])
                if record.home_off_line in ["PP_up", "PP_kill_dwn"]:
                    elo_list[record.home_team].penalty_lines_elo[record.home_off_line].update_player(
                        [away_def_pair_rating for _ in range(home_its)],
                        [away_def_pair_rd for _ in range(home_its)],
                        [False for _ in range(home_its)])
                else:
                    elo_list[record.home_team].off_lines_elo[record.home_off_line].update_player(
                        [away_def_pair_rating for _ in range(home_its)],
                        [away_def_pair_rd for _ in range(home_its)],
                        [False for _ in range(home_its)])

    return elo_list


# Main
if __name__ == "__main__":
    # Load games
    data = load_games("whl_2025.csv")
    print(f"Loaded {len(data)} games. \n\n")

    # Generate elos
    elos = generate_elo(data)
    print("Finished individual elo calculations.\n")

    # Inverse-variance weighting for overall metric
    overall_elos = {}
    for elo in elos.items():
        a = 1 / pow(elo[1].def_pairs_elo["first_def"].getRd(), 2)
        a += 1 / pow(elo[1].def_pairs_elo["second_def"].getRd(), 2)
        a += 1 / pow(elo[1].def_pairs_elo["empty_net_line"].getRd(), 2)
        a += 1 / pow(elo[1].off_lines_elo["first_off"].getRd(), 2)
        a += 1 / pow(elo[1].off_lines_elo["second_off"].getRd(), 2)
        a += 1 / pow(elo[1].off_lines_elo["empty_net_line"].getRd(), 2)
        a += 1 / pow(elo[1].penalty_lines_elo["PP_kill_dwn"].getRd(), 2)
        a += 1 / pow(elo[1].penalty_lines_elo["PP_up"].getRd(), 2)
        a += 1 / pow(elo[1].goalie_elo.getRd(), 2)
        final_var = 1 / a
        overall_elo = (1 / pow(elo[1].def_pairs_elo["first_def"].getRd(), 2)) / a * elo[1].def_pairs_elo["first_def"].getRating()
        overall_elo += (1 / pow(elo[1].def_pairs_elo["second_def"].getRd(), 2)) / a * elo[1].def_pairs_elo["second_def"].getRating()
        overall_elo += (1 / pow(elo[1].def_pairs_elo["empty_net_line"].getRd(), 2)) / a * elo[1].def_pairs_elo["empty_net_line"].getRating()
        overall_elo += (1 / pow(elo[1].off_lines_elo["first_off"].getRd(), 2)) / a * elo[1].off_lines_elo["first_off"].getRating()
        overall_elo += (1 / pow(elo[1].off_lines_elo["second_off"].getRd(), 2)) / a * elo[1].off_lines_elo["second_off"].getRating()
        overall_elo += (1 / pow(elo[1].off_lines_elo["empty_net_line"].getRd(), 2)) / a * elo[1].off_lines_elo["empty_net_line"].getRating()
        overall_elo += (1 / pow(elo[1].penalty_lines_elo["PP_kill_dwn"].getRd(), 2)) / a * elo[1].penalty_lines_elo["PP_kill_dwn"].getRating()
        overall_elo += (1 / pow(elo[1].penalty_lines_elo["PP_up"].getRd(), 2)) / a * elo[1].penalty_lines_elo["PP_up"].getRating()
        overall_elo += (1 / pow(elo[1].goalie_elo.getRd(), 2)) / a * elo[1].goalie_elo.getRating()

        overall_elos[elo[0]] = [overall_elo, final_var]
    overall_elos = sorted(overall_elos.items(), key=lambda x: x[1][0], reverse=True)
    for elo in overall_elos:
        print(elo[0] + ": " + str(elo[1][0]) + ", " + str(elo[1][1]))
    print()

    # Write results to csv file
    results = []
    cur = 1
    for elo in overall_elos:
        results.append({"rank": cur, "country": elo[0], "rating": elo[1][0], "rd (rating deviation)": elo[1][1]})
        cur += 1
    with open("power_ranking.csv", 'w', newline='') as f:
        fieldnames = ["rank", "country", "rating", "rd (rating deviation)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("Done! ")
