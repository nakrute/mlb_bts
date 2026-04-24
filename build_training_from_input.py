"""
build_training_from_input.py — Build training dataset from CSV input files
and train the model.

This script:
1. Reads historical data CSVs (batter stats, pitcher stats, games)
2. Constructs lagged feature rows for each player-game
3. Trains XGBoost classifier
4. Saves the model
"""
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

from config import DATA_DIR, MODEL_DIR, ROLLING_WINDOWS, MIN_AB_FOR_PLATOON, PARK_FACTORS, MIN_SEASON_PA, MIN_LINEUP_POSITION, MAX_LINEUP_POSITION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = os.path.join(DATA_DIR, "input")


def get_today_date_str():
    """Get today's date as YYYY-MM-DD string."""
    return datetime.today().strftime("%Y-%m-%d")


def get_pitcher_stats(pitcher_name: str, pitching_df: pd.DataFrame) -> dict:
    """
    Look up pitcher's current season stats.
    Returns ERA, WHIP for the pitcher.
    """
    if pd.isna(pitcher_name):
        return {"pitcher_era": 4.0, "pitcher_whip": 1.2}  # League average
    
    pitcher_name = str(pitcher_name).strip()
    # Find most recent season for this pitcher
    pitcher_data = pitching_df[pitching_df["name"].str.lower() == pitcher_name.lower()]
    if len(pitcher_data) == 0:
        return {"pitcher_era": 4.0, "pitcher_whip": 1.2}  # League average
    
    latest = pitcher_data.sort_values("season").iloc[-1]
    return {
        "pitcher_era": float(latest.get("era", 4.0)),
        "pitcher_whip": float(latest.get("whip", 1.2))
    }


def load_input_data():
    """Load all input CSV files based on today's date."""
    today_str = get_today_date_str()
    historical = pd.read_csv(os.path.join(INPUT_DIR, f"{today_str}_historical_data.csv"))
    pitching = pd.read_csv(os.path.join(INPUT_DIR, f"{today_str}_historical_pitching_data.csv"))
    players_games = pd.read_csv(os.path.join(INPUT_DIR, f"{today_str}_players_games.csv"))
    todays_games = pd.read_csv(os.path.join(INPUT_DIR, f"{today_str}_todays_games.csv"))
    
    # Create a game_id to pitcher mapping
    pitcher_map = {}
    for _, row in todays_games.iterrows():
        game_id = row["game_id"]
        home_team = row["home_name"]
        away_team = row["away_name"]
        pitcher_map[game_id] = {
            "home_pitcher": row["home_probable_pitcher"],
            "away_pitcher": row["away_probable_pitcher"],
            "home_team": home_team,
            "away_team": away_team
        }
    
    logger.info(f"Loaded {len(historical)} historical batter rows")
    logger.info(f"Loaded {len(pitching)} historical pitcher rows")
    logger.info(f"Loaded {len(players_games)} player-game mappings")
    logger.info(f"Loaded {len(todays_games)} today's games")
    
    return historical, pitching, players_games, todays_games, pitcher_map


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _hit_game_rate(frame: pd.DataFrame) -> float:
    if frame.empty or "hits" not in frame:
        return 0.0
    return frame["hits"].map(lambda x: 1 if _safe_float(x, 0) >= 1 else 0).mean()


def _lineup_quality(lineup_position: int) -> float:
    lineup_position = _safe_int(lineup_position, 0)
    if lineup_position <= 0:
        return 0.55
    return max(0.0, (10 - lineup_position) / 9.0)


def _build_row_from_prior(player_id, player_name, player_pos, current, prior,
                          pitcher_era=4.0, pitcher_whip=1.2, park_factor=1.0):
    """Build one row using only data available before the target row."""
    current_avg = _safe_float(current.get("avg", prior.iloc[-1].get("avg", 0.270) if len(prior) else 0.270), 0.270)
    current_hits = _safe_int(current.get("hits", 0))
    current_ab = _safe_int(current.get("atBats", 0))

    lineup_position = _safe_int(current.get("lineup_position", 0), 0)
    if lineup_position == 0:
        lineup_position = 5

    row = {
        "player_id": player_id,
        "player_name": player_name,
        "position": player_pos,
        "current_avg": current_avg,
        "current_hits": current_hits,
        "current_at_bats": current_ab,
        "lineup_position": lineup_position,
        "is_confirmed_starter": 1 if lineup_position > 0 else 0,
        "is_top_lineup": 1 if MIN_LINEUP_POSITION <= lineup_position <= 5 else 0,
        "lineup_quality": _lineup_quality(lineup_position),
        "batter_bats_left": _safe_int(current.get("batter_bats_left", 0), 0),
        "batter_bats_right": _safe_int(current.get("batter_bats_right", 0), 0),
        "batter_switch_hitter": _safe_int(current.get("batter_switch_hitter", 0), 0),
        "pitcher_throws_left": _safe_int(current.get("pitcher_throws_left", 0), 0),
        "platoon_advantage": _safe_int(current.get("platoon_advantage", 0), 0),
    }

    for window in [2, 3, 7, 14, 30]:
        window_prior = prior.tail(window)
        total_ab = window_prior["atBats"].map(lambda x: _safe_float(x, 0)).sum() if "atBats" in window_prior else 0
        total_hits = window_prior["hits"].map(lambda x: _safe_float(x, 0)).sum() if "hits" in window_prior else 0
        row[f"roll{window}_avg"] = total_hits / total_ab if total_ab > 0 else current_avg
        row[f"roll{window}_hit_game_rate"] = _hit_game_rate(window_prior) if len(window_prior) else current_avg
        row[f"roll{window}_ab_per_game"] = total_ab / len(window_prior) if len(window_prior) else current_ab

    prior_ab = prior["atBats"].map(lambda x: _safe_float(x, 0)).sum() if "atBats" in prior else 0
    prior_hits = prior["hits"].map(lambda x: _safe_float(x, 0)).sum() if "hits" in prior else 0

    row["season_pa"] = int(prior_ab)
    row["season_hits"] = int(prior_hits)
    row["season_avg"] = prior_hits / prior_ab if prior_ab > 0 else current_avg
    row["season_obp"] = _safe_float(prior.iloc[-1].get("obp", 0.320), 0.320) if len(prior) else 0.320
    row["season_ops"] = _safe_float(prior.iloc[-1].get("ops", 0.720), 0.720) if len(prior) else 0.720
    row["last_avg"] = _safe_float(prior.iloc[-1].get("avg", current_avg), current_avg) if len(prior) else current_avg
    row["avg_trending"] = 1 if len(prior) >= 2 and _safe_float(prior.iloc[-1].get("avg", 0), 0) > _safe_float(prior.iloc[0].get("avg", 0), 0) else 0
    row["last5_hit_games"] = int(prior.tail(5)["hits"].map(lambda x: 1 if _safe_float(x, 0) >= 1 else 0).sum()) if len(prior) and "hits" in prior else 0
    row["last10_hit_games"] = int(prior.tail(10)["hits"].map(lambda x: 1 if _safe_float(x, 0) >= 1 else 0).sum()) if len(prior) and "hits" in prior else 0

    streak = 0
    if len(prior) and "hits" in prior:
        for h in reversed(prior["hits"].tolist()):
            if _safe_float(h, 0) >= 1:
                streak += 1
            else:
                break
    row["current_streak"] = streak

    row["park_factor"] = float(park_factor)
    row["hitter_park_boost"] = 1 if park_factor > 1.02 else 0
    row["pitcher_era"] = pitcher_era
    row["pitcher_whip"] = pitcher_whip
    return row


def build_training_data(historical: pd.DataFrame, pitching: pd.DataFrame, players_games: pd.DataFrame, pitcher_map: dict):
    """
    Build training data. If per-game rows are present, target is actual hits>=1.
    Otherwise falls back to the uploaded season-level data while keeping the same feature schema.
    """
    all_rows = []
    batters = historical[historical["position"] != "P"].copy()
    logger.info(f"Filtered to {len(batters)} non-pitcher rows (removed {len(historical) - len(batters)} pitchers)")

    sort_cols = [c for c in ["game_date", "date", "game_id", "season"] if c in batters.columns] or ["season"]
    has_game_level_rows = any(c in batters.columns for c in ["game_date", "date", "game_id"])

    player_to_games = players_games.groupby("player_id")["game_id"].apply(list).to_dict() if not players_games.empty else {}

    for player_id, player_group in batters.groupby("id"):
        player_group = player_group.sort_values(sort_cols)
        player_name = player_group.iloc[0].get("name", str(player_id))
        player_pos = player_group.iloc[0].get("position", "DH")

        if len(player_group) < 3:
            continue

        pitcher_era, pitcher_whip = 4.0, 1.2
        player_games = player_to_games.get(player_id, [])
        if player_games and player_games[0] in pitcher_map:
            info = pitcher_map[player_games[0]]
            h_stats = get_pitcher_stats(info.get("home_pitcher"), pitching)
            a_stats = get_pitcher_stats(info.get("away_pitcher"), pitching)
            pitcher_era = (h_stats["pitcher_era"] + a_stats["pitcher_era"]) / 2.0
            pitcher_whip = (h_stats["pitcher_whip"] + a_stats["pitcher_whip"]) / 2.0

        avg_park_factor = np.mean(list(PARK_FACTORS.values())) / 100.0

        for i in range(2, len(player_group)):
            prior = player_group.iloc[:i]
            current = player_group.iloc[i]
            row = _build_row_from_prior(player_id, player_name, player_pos, current, prior, pitcher_era, pitcher_whip, avg_park_factor)
            row["got_hit"] = 1 if (_safe_float(current.get("hits", 0), 0) >= 1 if has_game_level_rows else row["current_avg"] >= 0.265) else 0
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("No training rows were built.")
        return df
    logger.info(f"Built {len(df)} training rows (position players only)")
    logger.info(f"Hit distribution: {df['got_hit'].value_counts().to_dict()}")
    if not has_game_level_rows:
        logger.warning("Training target is still season-level fallback. Upgrade train_model.py to fetch batter game logs for true BTS labels.")
    return df

def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix and target."""
    feature_cols = [
        "current_avg", "current_hits", "current_at_bats",
        "roll2_avg", "roll3_avg",
        "roll7_avg", "roll7_hit_game_rate", "roll7_ab_per_game",
        "roll14_avg", "roll14_hit_game_rate", "roll14_ab_per_game",
        "roll30_avg", "roll30_hit_game_rate", "roll30_ab_per_game",
        "season_pa", "season_hits", "season_avg", "season_obp", "season_ops",
        "last_avg", "avg_trending", "last5_hit_games", "last10_hit_games", "current_streak",
        "lineup_position", "is_confirmed_starter", "is_top_lineup", "lineup_quality",
        "batter_bats_left", "batter_bats_right", "batter_switch_hitter",
        "pitcher_throws_left", "platoon_advantage",
        "park_factor", "hitter_park_boost", "pitcher_era", "pitcher_whip",
    ]

    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].fillna(0)
    y = df["got_hit"].astype(int)

    logger.info(f"Using {len(available_features)} features: {available_features}")
    logger.info(f"Training on {len(X)} samples")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y, available_features

def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train XGBoost + calibrate."""
    base_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    
    # Cross-validation
    min_class = int(y.value_counts().min())
    n_splits = max(2, min(5, min_class))
    logger.info(f"Running {n_splits}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_auc = cross_val_score(base_model, X, y, cv=cv, scoring="roc_auc")
    logger.info(f"CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    
    # Calibrate
    logger.info("Calibrating probabilities...")
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=n_splits)
    calibrated.fit(X, y)
    
    # Evaluate
    probs = calibrated.predict_proba(X)[:, 1]
    brier = brier_score_loss(y, probs)
    auc = roc_auc_score(y, probs)
    logger.info(f"Training set — ROC-AUC: {auc:.4f}, Brier score: {brier:.4f}")
    
    preds = (probs >= 0.5).astype(int)
    logger.info("\n" + classification_report(y, preds, target_names=["No Hit", "Hit"]))
    
    return base_model, calibrated


def save_model(calibrated_model, feature_cols: list, base_model=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "bts_model.pkl")
    payload = {
        "model": calibrated_model,
        "feature_cols": feature_cols,
        "base_model": base_model,
        "trained_at": datetime.now().isoformat(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Model saved to {path}")


def main():
    logger.info("=" * 60)
    logger.info("BUILD TRAINING DATA FROM INPUT CSVs & TRAIN MODEL")
    logger.info("=" * 60)
    
    # Load input data
    historical, pitching, players_games, todays_games, pitcher_map = load_input_data()
    
    # Build training data
    df = build_training_data(historical, pitching, players_games, pitcher_map)
    
    if df.empty:
        logger.error("Failed to build training data")
        return
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Train model
    base_model, calibrated_model = train_model(X, y)
    
    # Save model
    save_model(calibrated_model, feature_cols, base_model)
    
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
