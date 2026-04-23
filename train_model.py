# train_model.py — Trains the XGBoost hit-probability classifier
"""
Usage:
    python train_model.py

This script:
1. Pulls multi-season game logs for a representative set of players
2. Builds feature rows for each game using historical data
3. Trains an XGBoost classifier calibrated to output true probabilities
4. Saves the model to models/bts_model.pkl

Run this once before using daily_picks.py. Re-train every few weeks during the season.
"""
import logging
import os
import pickle
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss, classification_report, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

from config import (
    TRAINING_START_YEAR, TRAINING_END_YEAR, MODEL_DIR,
    DATA_DIR, ROLLING_WINDOWS, PARK_FACTORS
)
from feature_engineer import get_feature_columns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load or build training dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_or_build_training_data() -> pd.DataFrame:
    """
    Loads cached training data if available, otherwise builds it from scratch.
    Building from scratch is slow (~hours for all players). Cache is recommended.
    """
    cache_path = os.path.join(DATA_DIR, "training_data.parquet")
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        logger.info(f"Loading cached training data from {cache_path}")
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded {len(df)} training rows")
        return df

    logger.info("No cached data found. Building training dataset...")
    logger.info("This will take 30-60 minutes. Get a coffee ☕")
    df = _build_training_dataset()
    df.to_parquet(cache_path, index=False)
    logger.info(f"Saved {len(df)} training rows to {cache_path}")
    return df


def _build_training_dataset() -> pd.DataFrame:
    """
    Builds a training dataset by fetching historical game logs for a broad
    set of players and constructing lag features for each game.
    """
    import statsapi
    from data_fetcher import get_historical_game_logs

    # Get a broad list of active/recent players
    # In practice you'd pull all batters who had significant PA each season
    # Here we get the current roster and pull their history
    all_rows = []

    for year in range(TRAINING_START_YEAR, TRAINING_END_YEAR + 1):
        logger.info(f"Pulling {year} season data...")
        try:
            # Get player IDs from live games in this season
            # More reliable than querying stats endpoint directly
            qualified_players = set()
            
            season_start = f"{year}-03-01"
            season_end = f"{year}-11-01"
            
            logger.info(f"  Fetching schedule from {season_start} to {season_end}...")
            try:
                schedule = statsapi.schedule(start_date=season_start, end_date=season_end)
                logger.info(f"  Found {len(schedule)} games")
                
                # Extract player IDs from a sample of games
                for game in schedule[:200]:  # Sample first 200 games
                    try:
                        boxscore = statsapi.boxscore_data(game['game_id'])
                        for side in ['home', 'away']:
                            players_dict = boxscore.get(side, {}).get('players', {})
                            for player_key in players_dict.keys():
                                # Player keys are like 'ID623197'
                                player_id_str = player_key.replace('ID', '')
                                if player_id_str.isdigit():
                                    qualified_players.add(int(player_id_str))
                        time.sleep(0.05)  # Rate limiting
                    except Exception as e:
                        logger.debug(f"  Error parsing boxscore: {e}")
                        pass
            except Exception as e:
                logger.warning(f"  Error fetching schedule for {year}: {e}")
            
            logger.info(f"  Found {len(qualified_players)} unique players from {len(schedule) if 'schedule' in locals() else 0} games in {year}")

            if not qualified_players:
                logger.warning(f"  No qualified players found for {year}, skipping...")
                continue

            for pid in list(qualified_players)[:100]:  # Process up to 100 players per year
                try:
                    game_log = get_historical_game_logs(pid, year, year)
                    if game_log.empty:
                        continue
                    
                    # Debug: check game_log structure
                    if 'got_hit' not in game_log.columns:
                        logger.debug(f"Player {pid} game_log missing 'got_hit' column. Columns: {list(game_log.columns)}")
                        continue
                    
                    if len(game_log) < 20:
                        logger.debug(f"Player {pid} has only {len(game_log)} games, need 20+")
                        continue

                    pname = f"Player_{pid}"  # Fallback name
                    rows = _build_lagged_rows(game_log, pid, pname, year)
                    if rows:
                        all_rows.extend(rows)
                        logger.debug(f"  Player {pid}: Built {len(rows)} lagged rows from {len(game_log)} games")
                except Exception as e:
                    logger.debug(f"  Error for player {pid}: {e}")

        except Exception as e:
            logger.error(f"Failed to pull {year} players: {e}")

    df = pd.DataFrame(all_rows)
    logger.info(f"Total training rows: {len(df)}")
    if not df.empty and 'got_hit' in df.columns:
        logger.info(f"Hit rate: {df['got_hit'].mean():.3f}")
    else:
        logger.warning("No training data collected. Check API connectivity and player data.")
    return df


def _build_lagged_rows(game_log: pd.DataFrame, player_id: int,
                        player_name: str, season: int) -> list[dict]:
    """
    For each game in a player's game log, compute lag features
    (rolling stats computed from PRIOR games only — no leakage).
    """
    rows = []
    game_log = game_log.sort_values("date").reset_index(drop=True)
    
    # Verify required columns exist
    required_cols = ["date", "got_hit", "ab", "hits", "so", "bb"]
    missing_cols = [col for col in required_cols if col not in game_log.columns]
    if missing_cols:
        logger.warning(f"Game log for {player_name} missing columns: {missing_cols}")
        return rows

    for i in range(20, len(game_log)):  # need at least 20 prior games
        prior = game_log.iloc[:i]
        current = game_log.iloc[i]

        row = {
            "player_id":   player_id,
            "player_name": player_name,
            "season":      season,
            "date":        current["date"],
            "got_hit":     int(current["got_hit"]),
        }

        for window in ROLLING_WINDOWS:
            recent = prior.tail(window)
            if recent.empty or recent["ab"].sum() == 0:
                continue
            total_ab   = recent["ab"].sum()
            total_hits = recent["hits"].sum()
            games_hit  = recent["got_hit"].sum()
            total_games = len(recent)
            row[f"roll{window}_avg"]           = total_hits / total_ab if total_ab > 0 else 0.250
            row[f"roll{window}_hit_game_rate"] = games_hit / total_games
            row[f"roll{window}_ab_per_game"]   = total_ab / total_games
            row[f"roll{window}_k_pct"]         = recent["so"].sum() / total_ab if total_ab > 0 else 0.20
            row[f"roll{window}_bb_pct"]        = recent["bb"].sum() / total_ab if total_ab > 0 else 0.08

        row["current_streak"]  = _streak(prior)
        row["last5_hit_games"] = int(prior.tail(5)["got_hit"].sum())
        row["last10_hit_games"]= int(prior.tail(10)["got_hit"].sum())

        # Season cumulative (using prior games only)
        season_prior = prior[prior["date"].dt.year == season]
        row["season_pa"]      = int(season_prior["ab"].sum())
        row["season_hits"]    = int(season_prior["hits"].sum())
        row["season_avg"]     = (
            season_prior["hits"].sum() / season_prior["ab"].sum()
            if season_prior["ab"].sum() > 0 else 0.250
        )
        row["season_games"]   = len(season_prior)

        rows.append(row)

    return rows


def _streak(game_log: pd.DataFrame) -> int:
    streak = 0
    for v in reversed(game_log["got_hit"].tolist()):
        if v == 1:
            streak += 1
        else:
            break
    return streak


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Train the model
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_FEATURES = [
    "roll7_avg", "roll7_hit_game_rate", "roll7_ab_per_game", "roll7_k_pct", "roll7_bb_pct",
    "roll14_avg", "roll14_hit_game_rate", "roll14_ab_per_game",
    "roll30_avg", "roll30_hit_game_rate", "roll30_ab_per_game",
    "current_streak", "last5_hit_games", "last10_hit_games",
    "season_avg", "season_pa", "season_games",
]


def train(df: pd.DataFrame) -> tuple:
    """Train XGBoost + calibrate. Returns (model, calibrated_model, feature_cols)."""
    if df.empty or 'got_hit' not in df.columns:
        raise ValueError(
            "Training dataset is empty or missing 'got_hit' column. "
            "Check API connectivity or adjust TRAINING_START_YEAR/TRAINING_END_YEAR config."
        )
    
    df = df.dropna(subset=["got_hit"])
    available_features = [f for f in TRAIN_FEATURES if f in df.columns]
    X = df[available_features].fillna(0)
    y = df["got_hit"].astype(int)

    logger.info(f"Training on {len(X)} samples, {len(available_features)} features")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    # Base XGBoost classifier
    base_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
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
    )

    # Cross-validation to gauge performance
    logger.info("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(base_model, X, y, cv=cv, scoring="roc_auc")
    logger.info(f"CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # Calibrate probabilities using isotonic regression
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    calibrated.fit(X, y)

    # Evaluate on full training set (in-sample, just for reference)
    probs = calibrated.predict_proba(X)[:, 1]
    brier = brier_score_loss(y, probs)
    auc   = roc_auc_score(y, probs)
    logger.info(f"Training set — ROC-AUC: {auc:.4f}, Brier score: {brier:.4f}")

    preds = (probs >= 0.5).astype(int)
    logger.info("\n" + classification_report(y, preds, target_names=["No Hit", "Hit"]))

    return base_model, calibrated, available_features


def save_model(calibrated_model, feature_cols: list, base_model=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "bts_model.pkl")
    payload = {
        "model":        calibrated_model,
        "feature_cols": feature_cols,
        "base_model":   base_model,
        "trained_at":   datetime.now().isoformat(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Model saved to {path}")


def load_model() -> tuple:
    path = os.path.join(MODEL_DIR, "bts_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at {path}. Run: python train_model.py"
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)
    logger.info(f"Loaded model trained at {payload['trained_at']}")
    return payload["model"], payload["feature_cols"]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("BTS MODEL — TRAINING PIPELINE")
    logger.info("=" * 60)

    df = load_or_build_training_data()
    base_model, calibrated_model, feature_cols = train(df)
    save_model(calibrated_model, feature_cols, base_model)

    logger.info("Training complete. You can now run: python daily_picks.py")