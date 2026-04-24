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

from config import DATA_DIR, MODEL_DIR, ROLLING_WINDOWS, MIN_AB_FOR_PLATOON, PARK_FACTORS

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


def build_training_data(historical: pd.DataFrame, pitching: pd.DataFrame, players_games: pd.DataFrame, pitcher_map: dict):
    """
    Build training dataset with lagged features from historical data.
    Filters out pitchers and focuses on position players.
    """
    all_rows = []
    
    # Filter out pitchers (position == "P")
    batters = historical[historical["position"] != "P"].copy()
    logger.info(f"Filtered to {len(batters)} non-pitcher rows (removed {len(historical) - len(batters)} pitchers)")
    
    # Get games for each player
    player_to_games = players_games.groupby("player_id")["game_id"].apply(list).to_dict()
    
    # Group by player to build game logs
    for player_id, player_group in batters.groupby("id"):
        player_group = player_group.sort_values("season")
        player_name = player_group.iloc[0]["name"]
        
        # Get games for this player
        player_games = player_to_games.get(player_id, [])
        player_pos = player_group.iloc[0]["position"]
        
        # For each season, build lagged features
        for season, season_group in player_group.groupby("season"):
            season_group = season_group.sort_values("season")
            
            # We need at least 3 historical points
            if len(season_group) < 3:
                continue
            
            # Build feature rows
            for i in range(2, len(season_group)):
                prior = season_group.iloc[:i]
                current = season_group.iloc[i]
                
                # Target: did player get a hit? Use average as proxy
                # Above .265 = likely hit, below that = likely no-hit
                avg = float(current.get("avg", 0.270))
                got_hit = 1 if avg >= 0.265 else 0
                
                row = {
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": player_pos,
                    "season": season,
                    "got_hit": got_hit,
                    "current_avg": avg,
                    "current_hits": int(current.get("hits", 0)),
                    "current_at_bats": int(current.get("atBats", 0)),
                }
                
                # Position scoring: early lineup positions are best (1-5 > 6-9)
                lineup_pos = ord(player_pos[0]) - ord('0') if player_pos and len(player_pos) > 0 else 9
                row["is_top_lineup"] = 1 if lineup_pos <= 5 else 0
                
                # Lagged rolling stats
                for window in [2, 3]:
                    window_prior = prior.tail(window)
                    if len(window_prior) > 0 and window_prior["atBats"].sum() > 0:
                        total_ab = window_prior["atBats"].sum()
                        total_hits = window_prior["hits"].sum()
                        row[f"roll{window}_avg"] = total_hits / total_ab if total_ab > 0 else 0.250
                    else:
                        row[f"roll{window}_avg"] = 0.250
                
                # Season cumulative
                row["season_pa"] = int(prior["atBats"].sum())
                row["season_hits"] = int(prior["hits"].sum())
                row["season_avg"] = row["season_hits"] / row["season_pa"] if row["season_pa"] > 0 else 0.250
                row["season_obp"] = float(prior.iloc[-1].get("obp", 0.320))
                row["season_ops"] = float(prior.iloc[-1].get("ops", 0.720))
                
                # Simple streak/consistency
                row["last_avg"] = float(prior.iloc[-1].get("avg", 0.270))
                row["avg_trending"] = 1 if prior.iloc[-1].get("avg", 0) > prior.iloc[0].get("avg", 0) else 0
                
                # Hitter park indicator (use average of all park factors for now as proxy)
                avg_park_factor = np.mean([v for v in PARK_FACTORS.values()]) / 100.0
                row["hitter_park_boost"] = 1.0 if avg_park_factor > 1.02 else 0.0  # 102+ is good
                
                # Pitcher matchup - find the opposing pitcher for this game
                opponent_pitcher = "Unknown"
                if player_id in player_to_games:
                    game_idx = 0  # Assume first game is the one we want (could be improved)
                if game_idx < len(player_games):
                    game_id = player_games[game_idx]
                    if game_id in pitcher_map:
                        # Determine if player's team is home or away
                        home_team = pitcher_map[game_id]["home_team"]
                        # For now, use average of both possible pitchers
                        h_pitcher = pitcher_map[game_id]["home_pitcher"]
                        a_pitcher = pitcher_map[game_id]["away_pitcher"]
                        h_stats = get_pitcher_stats(h_pitcher, pitching)
                        a_stats = get_pitcher_stats(a_pitcher, pitching)
                        row["pitcher_era"] = (h_stats["pitcher_era"] + a_stats["pitcher_era"]) / 2.0
                        row["pitcher_whip"] = (h_stats["pitcher_whip"] + a_stats["pitcher_whip"]) / 2.0
                    else:
                        row["pitcher_era"] = 4.0
                        row["pitcher_whip"] = 1.2
                else:
                    row["pitcher_era"] = 4.0
                    row["pitcher_whip"] = 1.2
                
                all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    logger.info(f"Built {len(df)} training rows (position players only)")
    logger.info(f"Hit distribution: {df['got_hit'].value_counts().to_dict()}")
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix and target."""
    feature_cols = [
        "current_avg", "current_hits", "current_at_bats",
        "roll2_avg", "roll3_avg",
        "season_pa", "season_hits", "season_avg", "season_obp", "season_ops",
        "last_avg", "avg_trending",
        "is_top_lineup", "hitter_park_boost",
        "pitcher_era", "pitcher_whip"
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
    logger.info("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(base_model, X, y, cv=cv, scoring="roc_auc")
    logger.info(f"CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    
    # Calibrate
    logger.info("Calibrating probabilities...")
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
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
