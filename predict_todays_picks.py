"""
predict_todays_picks.py — Generate today's top 2 BTS picks using the trained model

This script:
1. Loads today's games from input CSV
2. Matches players to historical data
3. Builds feature rows for each player
4. Predicts hit probabilities
5. Selects top 2 picks
6. Saves picks to CSV
"""
import logging
import os
import pickle
from datetime import datetime

import pandas as pd
import numpy as np

from config import DATA_DIR, MODEL_DIR, PICKS_DIR, TOP_N_PICKS, PARK_FACTORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = os.path.join(DATA_DIR, "input")


def get_today_date_str():
    """Get today's date as YYYY-MM-DD string."""
    return datetime.today().strftime("%Y-%m-%d")


def load_model():
    """Load trained model."""
    path = os.path.join(MODEL_DIR, "bts_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run build_training_from_input.py first.")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["feature_cols"]


def load_data():
    """Load input data files based on today's date."""
    today_str = get_today_date_str()
    historical = pd.read_csv(os.path.join(INPUT_DIR, f"{today_str}_historical_data.csv"))
    todays_games = pd.read_csv(os.path.join(INPUT_DIR, f"{today_str}_todays_games.csv"))
    players_games = pd.read_csv(os.path.join(INPUT_DIR, f"{today_str}_players_games.csv"))
    
    return historical, todays_games, players_games


def build_candidate_rows(historical: pd.DataFrame, todays_games: pd.DataFrame, 
                        players_games: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature rows for all players in today's games.
    Filters out pitchers and focuses on top-of-lineup hitters.
    """
    candidates = []
    
    # Filter out pitchers
    batters = historical[historical["position"] != "P"].copy()
    
    # Get all non-pitcher players in today's games
    today_player_ids = players_games["player_id"].unique()
    
    for player_id in today_player_ids:
        # Get player's most recent season data
        player_data = batters[batters["id"] == player_id].sort_values("season", ascending=False)
        
        if player_data.empty:
            continue
        
        # Use most recent season as features
        latest = player_data.iloc[0]
        player_pos = latest.get("position", "DH")
        
        # Build simple feature row
        row = {
            "player_id": player_id,
            "player_name": latest["name"],
            "position": player_pos,
            "current_avg": float(latest.get("avg", 0.270)),
            "current_hits": int(latest.get("hits", 0)),
            "current_at_bats": int(latest.get("atBats", 0)),
            "season_avg": float(latest.get("avg", 0.270)),
            "season_obp": float(latest.get("obp", 0.320)),
            "season_ops": float(latest.get("ops", 0.720)),
        }
        
        # Top-of-lineup indicator (positions 2B, SS, 1B, 3B, OF typically bat early)
        # Use position name to infer — could also use lineup_pos if available
        is_key_pos = player_pos.upper() in ["1B", "2B", "SS", "3B", "LF", "CF", "RF", "C", "DH"]
        row["is_top_lineup"] = 1 if is_key_pos else 0
        
        # Rolling averages
        prior_seasons = player_data.iloc[1:4] if len(player_data) > 1 else pd.DataFrame()
        if not prior_seasons.empty:
            row["roll2_avg"] = float(prior_seasons.iloc[0].get("avg", 0.270))
            row["roll3_avg"] = float(prior_seasons.iloc[1].get("avg", 0.270) if len(prior_seasons) > 1 else 0.270)
        else:
            row["roll2_avg"] = float(latest.get("avg", 0.270))
            row["roll3_avg"] = float(latest.get("avg", 0.270))
        
        # Season cumulative
        row["season_pa"] = int(latest.get("atBats", 0))
        row["season_hits"] = int(latest.get("hits", 0))
        row["last_avg"] = float(latest.get("avg", 0.270))
        row["avg_trending"] = 1  # Assume trending up
        
        # Map to today's game and get park info
        game_row = players_games[players_games["player_id"] == player_id]
        if not game_row.empty:
            game_id = game_row.iloc[0]["game_id"]
            game = todays_games[todays_games["game_id"] == game_id]
            if not game.empty:
                game_info = game.iloc[0]
                row["game_id"] = game_id
                row["away_team"] = game_info["away_name"]
                row["home_team"] = game_info["home_name"]
                row["game_datetime"] = game_info["game_datetime"]
                row["venue_name"] = game_info.get("venue_name", "")
                row["pitcher_era"] = float(game_info.get("pitcher_era", 4.50))
                row["pitcher_whip"] = float(game_info.get("pitcher_whip", 1.30))
                # Hitter park indicator (high park factor > 102)
                venue = row["venue_name"]
                park_factor = PARK_FACTORS.get(venue, 100) / 100.0
                row["hitter_park_boost"] = 1 if park_factor > 1.02 else 0
        else:
            row["hitter_park_boost"] = 0
        
        candidates.append(row)
    
    return pd.DataFrame(candidates)


def predict_picks(model, feature_cols: list, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Predict hit probabilities for all candidates.
    """
    # Ensure we have the exact features the model expects
    X = candidates[feature_cols].fillna(0)
    
    # Predict
    probs = model.predict_proba(X)[:, 1]
    candidates = candidates.reset_index(drop=True)
    candidates["p_hit"] = probs
    
    # Sort by probability and select top N
    top_picks = candidates.sort_values("p_hit", ascending=False).head(TOP_N_PICKS)
    
    return top_picks


def main():
    logger.info("=" * 60)
    logger.info("PREDICTING TODAY'S TOP BTS PICKS")
    logger.info("=" * 60)
    
    # Load model
    logger.info("Loading trained model...")
    model, feature_cols = load_model()
    logger.info(f"Model expects features: {feature_cols}")
    
    # Load data
    logger.info("Loading data...")
    historical, todays_games, players_games = load_data()
    
    # Build candidates
    logger.info("Building feature rows for today's candidates...")
    candidates = build_candidate_rows(historical, todays_games, players_games)
    logger.info(f"Built features for {len(candidates)} players")
    
    if candidates.empty:
        logger.warning("No candidates found. Exiting.")
        return
    
    # Predict
    logger.info(f"Predicting hit probabilities...")
    picks = predict_picks(model, feature_cols, candidates)
    
    # Display picks
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TOP {TOP_N_PICKS} PICKS FOR TODAY")
    logger.info(f"{'=' * 60}")
    
    for i, (_, row) in enumerate(picks.iterrows(), 1):
        logger.info(
            f"{i}. {row['player_name']} ({row['position']}) | "
            f"Game: {row['away_team']} @ {row['home_team']} | "
            f"P(Hit) = {row['p_hit']:.3f}"
        )
    
    # Save picks
    os.makedirs(PICKS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(PICKS_DIR, f"picks_{timestamp}.csv")
    
    output_cols = ["player_id", "player_name", "position", "away_team", "home_team", "game_datetime", "p_hit"]
    picks[output_cols].to_csv(out_path, index=False)
    logger.info(f"\nSaved picks to {out_path}")


if __name__ == "__main__":
    main()
