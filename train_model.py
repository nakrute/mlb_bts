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
import statsapi
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

from config import (
    TRAINING_START_YEAR, TRAINING_END_YEAR, MODEL_DIR,
    DATA_DIR, ROLLING_WINDOWS, PARK_FACTORS
)
from data_fetcher import get_historical_game_logs
from feature_engineer import get_feature_columns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_todays_games() -> pd.DataFrame:
    games = statsapi.schedule()
    return pd.DataFrame(games)  


def get_players_in_games(games_df: pd.DataFrame) -> set:
    player_ids = set()
    for _, game in games_df.iterrows():
        try:
            boxscore = statsapi.boxscore_data(game['game_id'])
            for side in ['home', 'away']:
                players_dict = boxscore.get(side, {}).get('players', {})
                for player_key in players_dict.keys():
                    player_id_str = player_key.replace('ID', '')
                    if player_id_str.isdigit():
                        player_ids.add(int(player_id_str))
            time.sleep(0.02)  # Rate limiting
        except Exception as e:
            logger.debug(f"Error parsing boxscore for game {game['game_id']}: {e}")
            pass
    return player_ids


def historical_data_by_today_players() -> pd.DataFrame:
    today_games = get_todays_games()
    player_ids = get_players_in_games(today_games)
    player_ids = list(player_ids)
    logger.info(f"Found {len(player_ids)} unique players in today's games")
    
    all_df = pd.DataFrame()
    for count, pid in enumerate(player_ids, start=1):
        try:
            logging.info(f"Processing player {pid} ({count}/{len(player_ids)})")
            rows = get_historical_game_logs(pid, TRAINING_START_YEAR, TRAINING_END_YEAR)
            if not rows.empty:
                all_df = pd.concat([all_df, rows], ignore_index=True)
        except Exception as e:
            logger.debug(f"Error fetching historical data for player {pid}: {e}")
            pass
    
    logger.info(f"Built historical dataset with {len(all_df)} rows for today's players")
    return all_df



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


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("BTS MODEL — TRAINING PIPELINE")
    logger.info("=" * 60)
    df = historical_data_by_today_players()
    print(df)
    """
    df = load_or_build_training_data()
    base_model, calibrated_model, feature_cols = train(df)
    save_model(calibrated_model, feature_cols, base_model)
    """
    logger.info("Training complete. You can now run: python daily_picks.py")