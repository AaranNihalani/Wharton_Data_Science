#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "Assets"
OUTPUT_DIR = ROOT / "Orlando_Predictions"

INPUT_GAME_FILE = ASSETS_DIR / "whl_2025.csv"
INPUT_MATCHUPS_FILE = ASSETS_DIR / "WHSDSC_Rnd1_matchups.xlsx"

POWER_RANKINGS_FILE = OUTPUT_DIR / "power_rankings.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "round1_predictions.csv"
REPORT_FILE = OUTPUT_DIR / "model_report.md"


ELO_SCALE = 400.0
ELO_INIT = 1500.0
ROLLING_WINDOW = 10
HOLDOUT_FRACTION = 0.20


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def logistic_prob_from_elo_diff(diff: float, scale: float = ELO_SCALE) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


def log_loss_binary(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier_score(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    return float(((y - p) ** 2).mean())


def auc_rank(y_true: Iterable[int], y_pred: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(y_pred), dtype=float)

    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(p)) + 1

    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    for group_idx, count in enumerate(counts):
        if count > 1:
            tie_pos = np.where(inverse == group_idx)[0]
            ranks[tie_pos] = ranks[tie_pos].mean()

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class LogisticModel:
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    feature_names: List[str]


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    epochs: int = 6000,
    learning_rate: float = 0.03,
    l2: float = 3e-4,
) -> LogisticModel:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0

    xs = (x - mean) / std
    xb = np.c_[np.ones(len(xs)), xs]

    w = np.zeros(xb.shape[1], dtype=float)
    for _ in range(epochs):
        p = sigmoid(xb @ w)
        gradient = xb.T @ (p - y) / len(y)
        gradient[1:] += l2 * w[1:]
        w -= learning_rate * gradient

    return LogisticModel(weights=w, mean=mean, std=std, feature_names=feature_names)


def predict_logistic(model: LogisticModel, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xs = (x - model.mean) / model.std
    xb = np.c_[np.ones(len(xs)), xs]
    return np.asarray(sigmoid(xb @ model.weights), dtype=float)


@dataclass
class BradleyTerryModel:
    ratings: np.ndarray
    home_bias: float
    team_to_idx: Dict[str, int]


def fit_bradley_terry(
    train_games: pd.DataFrame,
    teams: List[str],
    epochs: int = 6000,
    learning_rate: float = 0.07,
    l2: float = 2e-3,
) -> BradleyTerryModel:
    team_to_idx = {team: i for i, team in enumerate(teams)}
    home_idx = train_games["home_team"].map(team_to_idx).to_numpy()
    away_idx = train_games["away_team"].map(team_to_idx).to_numpy()
    y = train_games["home_win"].to_numpy(dtype=float)

    ratings = np.zeros(len(teams), dtype=float)
    home_bias = 0.0

    for _ in range(epochs):
        z = home_bias + ratings[home_idx] - ratings[away_idx]
        p = sigmoid(z)
        error = p - y

        grad_r = np.zeros_like(ratings)
        np.add.at(grad_r, home_idx, error)
        np.add.at(grad_r, away_idx, -error)

        grad_r = grad_r / len(train_games) + l2 * ratings
        grad_h = error.mean()

        ratings -= learning_rate * grad_r
        home_bias -= learning_rate * grad_h

        ratings -= ratings.mean()

    return BradleyTerryModel(ratings=ratings, home_bias=home_bias, team_to_idx=team_to_idx)


def predict_bradley_terry(model: BradleyTerryModel, games: pd.DataFrame) -> np.ndarray:
    home_idx = games["home_team"].map(model.team_to_idx).to_numpy()
    away_idx = games["away_team"].map(model.team_to_idx).to_numpy()
    z = model.home_bias + model.ratings[home_idx] - model.ratings[away_idx]
    return np.asarray(sigmoid(z), dtype=float)


def load_game_data(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)

    numeric_cols = [
        "went_ot",
        "toi",
        "home_assists",
        "home_shots",
        "home_xg",
        "home_max_xg",
        "home_goals",
        "away_assists",
        "away_shots",
        "away_xg",
        "away_max_xg",
        "away_goals",
        "home_penalties_committed",
        "home_penalty_minutes",
        "away_penalties_committed",
        "away_penalty_minutes",
    ]

    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)

    raw["game_num"] = (
        raw["game_id"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    )

    game_df = (
        raw.groupby("game_id", as_index=False)
        .agg(
            game_num=("game_num", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            went_ot=("went_ot", "first"),
            home_goals=("home_goals", "sum"),
            away_goals=("away_goals", "sum"),
            home_xg=("home_xg", "sum"),
            away_xg=("away_xg", "sum"),
            home_shots=("home_shots", "sum"),
            away_shots=("away_shots", "sum"),
            home_penalty_minutes=("home_penalty_minutes", "sum"),
            away_penalty_minutes=("away_penalty_minutes", "sum"),
            total_toi=("toi", "sum"),
        )
        .sort_values("game_num")
        .reset_index(drop=True)
    )

    game_df["home_win"] = (game_df["home_goals"] > game_df["away_goals"]).astype(int)

    return game_df


def _rolling_form_state(records: deque) -> Dict[str, float]:
    if not records:
        return {
            "xg_diff60": 0.0,
            "shot_diff60": 0.0,
            "goal_diff_pg": 0.0,
            "penalty_diff_pg": 0.0,
            "xg_pace60": 0.0,
            "games": 0,
        }

    xg_for = sum(x["xg_for"] for x in records)
    xg_against = sum(x["xg_against"] for x in records)
    shots_for = sum(x["shots_for"] for x in records)
    shots_against = sum(x["shots_against"] for x in records)
    goals_for = sum(x["goals_for"] for x in records)
    goals_against = sum(x["goals_against"] for x in records)
    pen_for = sum(x["pen_for"] for x in records)
    pen_against = sum(x["pen_against"] for x in records)
    toi = sum(x["toi"] for x in records)

    if toi <= 0:
        toi = 3600.0 * max(1, len(records))

    return {
        "xg_diff60": (xg_for - xg_against) * 3600.0 / toi,
        "shot_diff60": (shots_for - shots_against) * 3600.0 / toi,
        "goal_diff_pg": (goals_for - goals_against) / len(records),
        "penalty_diff_pg": (pen_for - pen_against) / len(records),
        "xg_pace60": (xg_for + xg_against) * 3600.0 / toi,
        "games": float(len(records)),
    }


def simulate_elo_and_features(
    games: pd.DataFrame,
    hfa: float,
    k: float,
    rolling_window: int = ROLLING_WINDOW,
    elo_scale: float = ELO_SCALE,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, float]]]:
    teams = sorted(set(games["home_team"]).union(set(games["away_team"])))
    elo = {team: ELO_INIT for team in teams}

    history = defaultdict(lambda: deque(maxlen=rolling_window))
    rows: List[Dict[str, float | int | str]] = []

    for row in games.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team

        home_form = _rolling_form_state(history[home_team])
        away_form = _rolling_form_state(history[away_team])

        elo_diff = (elo[home_team] + hfa) - elo[away_team]
        p_elo = logistic_prob_from_elo_diff(elo_diff, scale=elo_scale)

        rows.append(
            {
                "game_id": row.game_id,
                "game_num": int(row.game_num),
                "home_team": home_team,
                "away_team": away_team,
                "home_win": int(row.home_win),
                "went_ot": int(row.went_ot),
                "elo_diff": float(elo_diff),
                "p_elo": float(p_elo),
                "xg_diff60_diff": float(home_form["xg_diff60"] - away_form["xg_diff60"]),
                "shot_diff60_diff": float(home_form["shot_diff60"] - away_form["shot_diff60"]),
                "goal_diff_pg_diff": float(home_form["goal_diff_pg"] - away_form["goal_diff_pg"]),
                "penalty_diff_pg_diff": float(
                    home_form["penalty_diff_pg"] - away_form["penalty_diff_pg"]
                ),
                "xg_pace60_diff": float(home_form["xg_pace60"] - away_form["xg_pace60"]),
                "form_games_min": int(min(home_form["games"], away_form["games"])),
            }
        )

        result = 1.0 if row.home_win == 1 else 0.0
        delta = k * (result - p_elo)
        elo[home_team] += delta
        elo[away_team] -= delta

        history[home_team].append(
            {
                "xg_for": float(row.home_xg),
                "xg_against": float(row.away_xg),
                "shots_for": float(row.home_shots),
                "shots_against": float(row.away_shots),
                "goals_for": float(row.home_goals),
                "goals_against": float(row.away_goals),
                "pen_for": float(row.home_penalty_minutes),
                "pen_against": float(row.away_penalty_minutes),
                "toi": float(row.total_toi),
            }
        )
        history[away_team].append(
            {
                "xg_for": float(row.away_xg),
                "xg_against": float(row.home_xg),
                "shots_for": float(row.away_shots),
                "shots_against": float(row.home_shots),
                "goals_for": float(row.away_goals),
                "goals_against": float(row.home_goals),
                "pen_for": float(row.away_penalty_minutes),
                "pen_against": float(row.home_penalty_minutes),
                "toi": float(row.total_toi),
            }
        )

    final_form = {team: _rolling_form_state(history[team]) for team in teams}
    features = pd.DataFrame(rows).sort_values("game_num").reset_index(drop=True)
    return features, elo, final_form


def tune_elo(games: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    hfa_candidates = [0.0, 15.0, 25.0, 35.0, 45.0, 55.0, 70.0, 90.0]
    k_candidates = [4.0, 6.0, 8.0, 10.0, 14.0, 20.0, 28.0]

    split_idx = int(len(games) * (1.0 - HOLDOUT_FRACTION))
    y_test = games["home_win"].iloc[split_idx:].to_numpy(dtype=float)

    results: List[Dict[str, float]] = []
    for hfa in hfa_candidates:
        for k in k_candidates:
            feat, _, _ = simulate_elo_and_features(games, hfa=hfa, k=k)
            p_test = feat["p_elo"].iloc[split_idx:].to_numpy(dtype=float)

            results.append(
                {
                    "hfa": hfa,
                    "k": k,
                    "log_loss": log_loss_binary(y_test, p_test),
                    "brier": brier_score(y_test, p_test),
                    "auc": auc_rank(y_test.astype(int), p_test),
                }
            )

    tuning_df = pd.DataFrame(results).sort_values(["log_loss", "brier"]).reset_index(drop=True)
    best = tuning_df.iloc[0]
    best_params = {"hfa": float(best["hfa"]), "k": float(best["k"])}
    return best_params, tuning_df


def evaluate_models(
    games: pd.DataFrame,
    best_elo_params: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    split_idx = int(len(games) * (1.0 - HOLDOUT_FRACTION))

    features, _, _ = simulate_elo_and_features(
        games,
        hfa=best_elo_params["hfa"],
        k=best_elo_params["k"],
    )

    train_games = games.iloc[:split_idx].copy()
    test_games = games.iloc[split_idx:].copy()

    train_feat = features.iloc[:split_idx].copy()
    test_feat = features.iloc[split_idx:].copy()

    y_train = train_feat["home_win"].to_numpy(dtype=float)
    y_test = test_feat["home_win"].to_numpy(dtype=float)

    baseline_prob = np.repeat(y_train.mean(), len(test_feat))
    elo_prob = test_feat["p_elo"].to_numpy(dtype=float)

    teams = sorted(set(games["home_team"]).union(set(games["away_team"])))
    bt_model = fit_bradley_terry(train_games, teams=teams)
    bt_prob = predict_bradley_terry(bt_model, test_games)

    blend_features = [
        "elo_diff",
        "xg_diff60_diff",
        "shot_diff60_diff",
        "goal_diff_pg_diff",
        "penalty_diff_pg_diff",
        "xg_pace60_diff",
    ]

    logistic_model = fit_logistic_regression(
        train_feat[blend_features].to_numpy(dtype=float),
        y_train,
        feature_names=blend_features,
    )
    blend_prob = predict_logistic(logistic_model, test_feat[blend_features].to_numpy(dtype=float))

    metrics_rows = []
    for model_name, probs in [
        ("baseline", baseline_prob),
        ("elo", elo_prob),
        ("bradley_terry", bt_prob),
        ("elo_logistic_blend", blend_prob),
    ]:
        metrics_rows.append(
            {
                "model": model_name,
                "log_loss": log_loss_binary(y_test, probs),
                "brier": brier_score(y_test, probs),
                "auc": auc_rank(y_test.astype(int), probs),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["log_loss", "brier"]).reset_index(drop=True)

    elo_ll = float(metrics_df.loc[metrics_df["model"] == "elo", "log_loss"].iloc[0])
    best_row = metrics_df.iloc[0]
    best_model = str(best_row["model"])

    if best_model != "elo" and (elo_ll - float(best_row["log_loss"])) < 0.002:
        best_model = "elo"

    trained_models: Dict[str, object] = {
        "feature_columns": blend_features,
        "bt_train_model": bt_model,
        "blend_train_model": logistic_model,
        "best_model": best_model,
    }

    return metrics_df, trained_models


def fit_final_models(
    games: pd.DataFrame,
    best_elo_params: Dict[str, float],
    feature_columns: List[str],
) -> Dict[str, object]:
    features, final_elo, final_form = simulate_elo_and_features(
        games,
        hfa=best_elo_params["hfa"],
        k=best_elo_params["k"],
    )

    y = features["home_win"].to_numpy(dtype=float)

    blend_model = fit_logistic_regression(
        features[feature_columns].to_numpy(dtype=float),
        y,
        feature_names=feature_columns,
    )

    teams = sorted(set(games["home_team"]).union(set(games["away_team"])))
    bt_model = fit_bradley_terry(games, teams=teams)

    return {
        "features": features,
        "final_elo": final_elo,
        "final_form": final_form,
        "blend_model": blend_model,
        "bt_model": bt_model,
    }


def build_power_rankings(
    final_elo: Dict[str, float],
    final_form: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for team, elo in final_elo.items():
        form = final_form[team]
        rows.append(
            {
                "team": team,
                "elo": float(elo),
                "recent_xg_diff60": float(form["xg_diff60"]),
                "recent_shot_diff60": float(form["shot_diff60"]),
                "recent_goal_diff_pg": float(form["goal_diff_pg"]),
            }
        )

    ranking_df = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1

    ranking_df = ranking_df[
        ["rank", "team", "elo", "recent_xg_diff60", "recent_shot_diff60", "recent_goal_diff_pg"]
    ]

    return ranking_df


def _confidence_tier(prob: float) -> str:
    p = max(prob, 1.0 - prob)
    if p >= 0.75:
        return "high"
    if p >= 0.62:
        return "medium"
    return "low"


def build_matchup_predictions(
    matchups: pd.DataFrame,
    best_model_name: str,
    best_elo_params: Dict[str, float],
    final_elo: Dict[str, float],
    final_form: Dict[str, Dict[str, float]],
    feature_columns: List[str],
    blend_model: LogisticModel,
    bt_model: BradleyTerryModel,
) -> pd.DataFrame:
    output_rows = []

    for row in matchups.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team

        home_elo = final_elo[home_team]
        away_elo = final_elo[away_team]

        home_form = final_form[home_team]
        away_form = final_form[away_team]

        elo_diff = (home_elo + best_elo_params["hfa"]) - away_elo
        p_elo = logistic_prob_from_elo_diff(elo_diff)

        blend_feature_map = {
            "elo_diff": elo_diff,
            "xg_diff60_diff": home_form["xg_diff60"] - away_form["xg_diff60"],
            "shot_diff60_diff": home_form["shot_diff60"] - away_form["shot_diff60"],
            "goal_diff_pg_diff": home_form["goal_diff_pg"] - away_form["goal_diff_pg"],
            "penalty_diff_pg_diff": home_form["penalty_diff_pg"] - away_form["penalty_diff_pg"],
            "xg_pace60_diff": home_form["xg_pace60"] - away_form["xg_pace60"],
        }

        x_blend = np.array([[blend_feature_map[col] for col in feature_columns]], dtype=float)
        p_blend = float(predict_logistic(blend_model, x_blend)[0])

        bt_home_idx = bt_model.team_to_idx[home_team]
        bt_away_idx = bt_model.team_to_idx[away_team]
        p_bt = float(
            sigmoid(bt_model.home_bias + bt_model.ratings[bt_home_idx] - bt_model.ratings[bt_away_idx])
        )

        if best_model_name == "elo":
            p_home = p_elo
        elif best_model_name == "bradley_terry":
            p_home = p_bt
        elif best_model_name == "elo_logistic_blend":
            p_home = p_blend
        else:
            p_home = p_elo

        favorite = home_team if p_home >= 0.5 else away_team
        favorite_prob = p_home if p_home >= 0.5 else 1.0 - p_home

        output_rows.append(
            {
                "game": int(row.game),
                "game_id": row.game_id,
                "home_team": home_team,
                "away_team": away_team,
                "p_home_win": float(p_home),
                "favorite": favorite,
                "favorite_win_prob": float(favorite_prob),
                "confidence_tier": _confidence_tier(float(p_home)),
                "model_used": best_model_name,
                "p_home_win_elo": float(p_elo),
                "p_home_win_blend": float(p_blend),
                "p_home_win_bt": float(p_bt),
            }
        )

    pred_df = pd.DataFrame(output_rows).sort_values("game").reset_index(drop=True)
    return pred_df


def build_report(
    games: pd.DataFrame,
    best_elo_params: Dict[str, float],
    elo_tuning: pd.DataFrame,
    metrics_df: pd.DataFrame,
    best_model_name: str,
    rankings: pd.DataFrame,
    predictions: pd.DataFrame,
) -> str:
    def to_markdown_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        separator = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = []
        for _, row in df.iterrows():
            body.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join([header, separator] + body)

    n_games = len(games)
    n_teams = len(set(games["home_team"]).union(set(games["away_team"])))
    home_win_rate = float(games["home_win"].mean())
    ot_rate = float(games["went_ot"].mean())

    top_tuning = elo_tuning.head(10).copy()
    top_tuning["hfa"] = top_tuning["hfa"].map(lambda x: f"{x:.1f}")
    top_tuning["k"] = top_tuning["k"].map(lambda x: f"{x:.1f}")
    top_tuning["log_loss"] = top_tuning["log_loss"].map(lambda x: f"{x:.4f}")
    top_tuning["brier"] = top_tuning["brier"].map(lambda x: f"{x:.4f}")
    top_tuning["auc"] = top_tuning["auc"].map(lambda x: f"{x:.4f}")

    metric_table = metrics_df.copy()
    metric_table["log_loss"] = metric_table["log_loss"].map(lambda x: f"{x:.4f}")
    metric_table["brier"] = metric_table["brier"].map(lambda x: f"{x:.4f}")
    metric_table["auc"] = metric_table["auc"].map(lambda x: f"{x:.4f}")

    top5 = rankings.head(5)[["rank", "team", "elo"]].copy()
    top5["elo"] = top5["elo"].map(lambda x: f"{x:.1f}")

    pred_view = predictions[
        ["game", "home_team", "away_team", "p_home_win", "favorite", "favorite_win_prob"]
    ].copy()
    pred_view["p_home_win"] = pred_view["p_home_win"].map(lambda x: f"{x:.3f}")
    pred_view["favorite_win_prob"] = pred_view["favorite_win_prob"].map(lambda x: f"{x:.3f}")

    report = []
    report.append("# WHSDSC 2026 Quantitative Pipeline Report")
    report.append("")
    report.append("## Data Summary")
    report.append(f"- Games: {n_games}")
    report.append(f"- Teams: {n_teams}")
    report.append(f"- Home win rate: {home_win_rate:.3f}")
    report.append(f"- Overtime rate: {ot_rate:.3f}")
    report.append("")
    report.append("## Modeling Workflow")
    report.append("- Aggregated shift-level rows to game-level outcomes and totals.")
    report.append("- Engineered leakage-safe pregame form features (rolling window = 10 games).")
    report.append("- Tuned Elo (HFA, K) on a chronological holdout objective (log loss).")
    report.append("- Benchmarked Elo against Bradley-Terry and Elo+logistic feature blend.")
    report.append("")
    report.append("## Elo Tuning Top 10")
    report.append(to_markdown_table(top_tuning))
    report.append("")
    report.append("## Holdout Performance (Last 20 percent of season)")
    report.append(to_markdown_table(metric_table))
    report.append("")
    report.append("## Selected Model")
    report.append(f"- Model chosen for matchup predictions: `{best_model_name}`")
    report.append(
        f"- Tuned Elo hyperparameters: `HFA={best_elo_params['hfa']:.1f}`, `K={best_elo_params['k']:.1f}`, `scale=400`"
    )
    report.append("")
    report.append("## Top 5 Power Rankings")
    report.append(to_markdown_table(top5))
    report.append("")
    report.append("## Round 1 Matchup Predictions")
    report.append(to_markdown_table(pred_view))
    report.append("")
    report.append("## Output Files")
    report.append(f"- `{POWER_RANKINGS_FILE.relative_to(ROOT)}`")
    report.append(f"- `{PREDICTIONS_FILE.relative_to(ROOT)}`")
    report.append(f"- `{REPORT_FILE.relative_to(ROOT)}`")

    return "\n".join(report) + "\n"


def run_pipeline() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    games = load_game_data(INPUT_GAME_FILE)

    best_elo_params, elo_tuning = tune_elo(games)
    metrics_df, model_state = evaluate_models(games, best_elo_params)

    final_models = fit_final_models(
        games,
        best_elo_params,
        feature_columns=model_state["feature_columns"],
    )

    rankings = build_power_rankings(
        final_elo=final_models["final_elo"],
        final_form=final_models["final_form"],
    )

    matchups = pd.read_excel(INPUT_MATCHUPS_FILE)
    predictions = build_matchup_predictions(
        matchups=matchups,
        best_model_name=model_state["best_model"],
        best_elo_params=best_elo_params,
        final_elo=final_models["final_elo"],
        final_form=final_models["final_form"],
        feature_columns=model_state["feature_columns"],
        blend_model=final_models["blend_model"],
        bt_model=final_models["bt_model"],
    )

    rankings.to_csv(POWER_RANKINGS_FILE, index=False)
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    report = build_report(
        games=games,
        best_elo_params=best_elo_params,
        elo_tuning=elo_tuning,
        metrics_df=metrics_df,
        best_model_name=model_state["best_model"],
        rankings=rankings,
        predictions=predictions,
    )
    REPORT_FILE.write_text(report, encoding="utf-8")

    print("Pipeline complete.")
    print(f"Best Elo params: HFA={best_elo_params['hfa']:.1f}, K={best_elo_params['k']:.1f}")
    print("Holdout metrics:")
    print(metrics_df.to_string(index=False))
    print(f"Selected model: {model_state['best_model']}")
    print(f"Wrote {POWER_RANKINGS_FILE.relative_to(ROOT)}")
    print(f"Wrote {PREDICTIONS_FILE.relative_to(ROOT)}")
    print(f"Wrote {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    run_pipeline()
