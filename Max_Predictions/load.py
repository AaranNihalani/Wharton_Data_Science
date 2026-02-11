# Packages
from dataclasses import dataclass, field
import pandas as pd
from typing import List


@dataclass
class GameRecord:
    """One row / state within a game"""
    toi: int
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    went_ot: bool
    home_off_line: str
    home_def_pairing: str
    away_off_line: str
    away_def_pairing: str
    home_goalie: str
    away_goalie: str
    home_assists: int
    home_shots: int
    home_xg: float
    home_max_xg: float
    away_assists: int
    away_shots: int
    away_xg: float
    away_max_xg: float
    home_penalties_committed: int
    home_penalty_minutes: float
    away_penalties_committed: int
    away_penalty_minutes: float


@dataclass
class Game:
    """All data associated with a single game"""
    game_id: str
    records: List[GameRecord] = field(default_factory=list)

    def add_record(self, record: GameRecord):
        self.records.append(record)

def load_games(csv_path: str) -> List[Game]:
    """Load games from CSV."""
    df = pd.read_csv(csv_path)

    games: List[Game] = []
    game = None
    cur_game = None
    for _, row in df.iterrows():
        game_id = row["game_id"]
        if game_id != cur_game:
            if game is not None:
                games.append(game)
            game = Game(game_id)
            cur_game = game_id

        record = GameRecord(
            toi = int(row["toi"]),
            home_team = row["home_team"],
            away_team = row["away_team"],
            home_goals = int(row["home_goals"]),
            away_goals = int(row["away_goals"]),
            went_ot = bool(row["went_ot"]),
            home_off_line = row["home_off_line"],
            home_def_pairing = row["home_def_pairing"],
            away_off_line = row["away_off_line"],
            away_def_pairing = row["away_def_pairing"],
            home_goalie = row["home_goalie"],
            away_goalie = row["away_goalie"],
            home_assists = int(row["home_assists"]),
            home_shots = row["home_shots"],
            home_xg = float(row["home_xg"]),
            home_max_xg = float(row["home_max_xg"]),
            away_assists = int(row["away_assists"]),
            away_shots = row["away_shots"],
            away_xg = float(row["away_xg"]),
            away_max_xg = float(row["away_max_xg"]),
            home_penalties_committed = int(row["home_penalties_committed"]),
            home_penalty_minutes = float(row["home_penalty_minutes"]),
            away_penalties_committed=int(row["away_penalties_committed"]),
            away_penalty_minutes=float(row["away_penalty_minutes"])
        )
        game.add_record(record)

    return games
