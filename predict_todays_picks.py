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

from config import DATA_DIR, MODEL_DIR, PICKS_DIR, TOP_N_PICKS, PARK_FACTORS, TEAM_ABBREVS, MIN_SEASON_PA, MIN_HIT_PROBABILITY, MIN_LINEUP_POSITION, MAX_LINEUP_POSITION

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
    historical = historical.sort_values([c for c in ["id", "season", "game_id"] if c in historical.columns]).reset_index(drop=True)
    todays_games = todays_games.sort_values([c for c in ["game_id"] if c in todays_games.columns]).reset_index(drop=True)
    players_games = players_games.sort_values([c for c in ["player_id", "game_id"] if c in players_games.columns]).reset_index(drop=True)
    
    return historical, todays_games, players_games


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


def _team_abbrev(team_name: str) -> str:
    return TEAM_ABBREVS.get(str(team_name), str(team_name))


def _park_factor_for_game(game_info: pd.Series) -> float:
    """PARK_FACTORS is keyed by team abbreviation, so use the home team instead of venue name."""
    home_abbrev = _team_abbrev(game_info.get("home_name", ""))
    return PARK_FACTORS.get(home_abbrev, 100) / 100.0


def _lineup_quality(lineup_position: int) -> float:
    lineup_position = _safe_int(lineup_position, 0)
    if lineup_position <= 0:
        return 0.55
    return max(0.0, (10 - lineup_position) / 9.0)


def _handedness_features(source, pitcher_throws_left: int) -> dict:
    vs_lhp_avg = _safe_float(source.get("batter_vs_lhp_avg", 0.270), 0.270)
    vs_lhp_ops = _safe_float(source.get("batter_vs_lhp_ops", 0.720), 0.720)
    vs_lhp_pa = _safe_int(source.get("batter_vs_lhp_pa", 0), 0)
    vs_lhp_k_pct = _safe_float(source.get("batter_vs_lhp_k_pct", 0.22), 0.22)
    vs_lhp_bb_pct = _safe_float(source.get("batter_vs_lhp_bb_pct", 0.08), 0.08)
    vs_rhp_avg = _safe_float(source.get("batter_vs_rhp_avg", 0.270), 0.270)
    vs_rhp_ops = _safe_float(source.get("batter_vs_rhp_ops", 0.720), 0.720)
    vs_rhp_pa = _safe_int(source.get("batter_vs_rhp_pa", 0), 0)
    vs_rhp_k_pct = _safe_float(source.get("batter_vs_rhp_k_pct", 0.22), 0.22)
    vs_rhp_bb_pct = _safe_float(source.get("batter_vs_rhp_bb_pct", 0.08), 0.08)

    if pitcher_throws_left:
        matchup_avg, matchup_ops, matchup_pa = vs_lhp_avg, vs_lhp_ops, vs_lhp_pa
        matchup_k_pct, matchup_bb_pct = vs_lhp_k_pct, vs_lhp_bb_pct
    else:
        matchup_avg, matchup_ops, matchup_pa = vs_rhp_avg, vs_rhp_ops, vs_rhp_pa
        matchup_k_pct, matchup_bb_pct = vs_rhp_k_pct, vs_rhp_bb_pct

    return {
        "batter_vs_lhp_avg": vs_lhp_avg,
        "batter_vs_lhp_ops": vs_lhp_ops,
        "batter_vs_lhp_pa": vs_lhp_pa,
        "batter_vs_lhp_k_pct": vs_lhp_k_pct,
        "batter_vs_lhp_bb_pct": vs_lhp_bb_pct,
        "batter_vs_rhp_avg": vs_rhp_avg,
        "batter_vs_rhp_ops": vs_rhp_ops,
        "batter_vs_rhp_pa": vs_rhp_pa,
        "batter_vs_rhp_k_pct": vs_rhp_k_pct,
        "batter_vs_rhp_bb_pct": vs_rhp_bb_pct,
        "matchup_split_avg": matchup_avg,
        "matchup_split_ops": matchup_ops,
        "matchup_split_pa": matchup_pa,
        "matchup_split_k_pct": matchup_k_pct,
        "matchup_split_bb_pct": matchup_bb_pct,
    }


def _add_recent_form(row: dict, player_data: pd.DataFrame, latest: pd.Series) -> None:
    """Adds rolling features. With season-level input this is a fallback; with game logs it becomes true recent form."""
    current_avg = _safe_float(latest.get("avg", 0.270), 0.270)
    current_ab = _safe_int(latest.get("atBats", 0), 0)

    for window in [2, 3, 7, 14, 30]:
        prior = player_data.iloc[1:window + 1] if len(player_data) > 1 else pd.DataFrame()
        total_ab = prior["atBats"].map(lambda x: _safe_float(x, 0)).sum() if not prior.empty and "atBats" in prior else 0
        total_hits = prior["hits"].map(lambda x: _safe_float(x, 0)).sum() if not prior.empty and "hits" in prior else 0
        row[f"roll{window}_avg"] = total_hits / total_ab if total_ab > 0 else current_avg
        row[f"roll{window}_hit_game_rate"] = prior["hits"].map(lambda x: 1 if _safe_float(x, 0) >= 1 else 0).mean() if not prior.empty and "hits" in prior else current_avg
        row[f"roll{window}_ab_per_game"] = total_ab / len(prior) if len(prior) else current_ab

    row["last5_hit_games"] = int(player_data.iloc[1:6]["hits"].map(lambda x: 1 if _safe_float(x, 0) >= 1 else 0).sum()) if len(player_data) > 1 and "hits" in player_data else 0
    row["last10_hit_games"] = int(player_data.iloc[1:11]["hits"].map(lambda x: 1 if _safe_float(x, 0) >= 1 else 0).sum()) if len(player_data) > 1 and "hits" in player_data else 0

    streak = 0
    if len(player_data) > 1 and "hits" in player_data:
        for h in player_data.iloc[1:]["hits"].tolist():
            if _safe_float(h, 0) >= 1:
                streak += 1
            else:
                break
    row["current_streak"] = streak


def _season_snapshot(player_data: pd.DataFrame, latest: pd.Series) -> dict:
    if "game_date" in player_data.columns:
        latest_season = _safe_int(latest.get("season", 0), 0)
        season_rows = player_data[player_data["season"].map(lambda x: _safe_int(x, 0)) == latest_season]
        total_ab = season_rows["atBats"].map(lambda x: _safe_float(x, 0)).sum() if "atBats" in season_rows else 0
        total_hits = season_rows["hits"].map(lambda x: _safe_float(x, 0)).sum() if "hits" in season_rows else 0
        return {
            "current_avg": total_hits / total_ab if total_ab > 0 else 0.270,
            "current_hits": int(total_hits),
            "current_at_bats": int(total_ab),
            "season_avg": total_hits / total_ab if total_ab > 0 else 0.270,
            "season_pa": int(total_ab),
            "season_hits": int(total_hits),
            "season_obp": _safe_float(latest.get("obp", 0.320), 0.320),
            "season_ops": _safe_float(latest.get("ops", 0.720), 0.720),
            "last_avg": _safe_float(latest.get("avg", 0.270), 0.270),
        }

    season_pa = _safe_int(latest.get("atBats", 0), 0)
    return {
        "current_avg": _safe_float(latest.get("avg", 0.270), 0.270),
        "current_hits": _safe_int(latest.get("hits", 0), 0),
        "current_at_bats": season_pa,
        "season_avg": _safe_float(latest.get("avg", 0.270), 0.270),
        "season_obp": _safe_float(latest.get("obp", 0.320), 0.320),
        "season_ops": _safe_float(latest.get("ops", 0.720), 0.720),
        "season_pa": season_pa,
        "season_hits": _safe_int(latest.get("hits", 0), 0),
        "last_avg": _safe_float(latest.get("avg", 0.270), 0.270),
    }


def build_candidate_rows(historical: pd.DataFrame, todays_games: pd.DataFrame,
                        players_games: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature rows for all players in today's games.
    Filters out pitchers and low-sample candidates.
    """
    candidates = []
    batters = historical[historical["position"] != "P"].copy()

    today_player_ids = sorted(players_games["player_id"].dropna().unique())

    for player_id in today_player_ids:
        sort_cols = [c for c in ["game_date", "game_id", "season"] if c in batters.columns] or ["season"]
        player_data = batters[batters["id"] == player_id].sort_values(sort_cols, ascending=False)

        if player_data.empty:
            continue

        latest = player_data.iloc[0]
        player_pos = latest.get("position", "DH")
        season_stats = _season_snapshot(player_data, latest)
        season_pa = season_stats["season_pa"]

        if season_pa < MIN_SEASON_PA:
            continue

        row = {
            "player_id": player_id,
            "player_name": latest["name"],
            "position": player_pos,
            **season_stats,
            "avg_trending": 1 if len(player_data) > 1 and _safe_float(latest.get("avg", 0), 0) >= _safe_float(player_data.iloc[1].get("avg", 0), 0) else 0,
        }

        game_row = players_games[players_games["player_id"] == player_id]
        lineup_position = 0
        if not game_row.empty:
            lineup_position = _safe_int(game_row.iloc[0].get("lineup_position", 0), 0)

        # If confirmed lineups are unavailable, keep candidates but treat them neutrally.
        row["lineup_position"] = lineup_position
        row["is_confirmed_starter"] = 1 if lineup_position > 0 else 0
        row["is_top_lineup"] = 1 if MIN_LINEUP_POSITION <= lineup_position <= 5 else 0
        row["lineup_quality"] = _lineup_quality(lineup_position)

        for col in ["batter_bats_left", "batter_bats_right", "batter_switch_hitter", "pitcher_throws_left", "platoon_advantage"]:
            row[col] = _safe_int(game_row.iloc[0].get(col, 0), 0) if not game_row.empty else 0
        row["batter_bat_side"] = game_row.iloc[0].get("batter_bat_side", "") if not game_row.empty else ""
        row["opponent_pitcher_hand"] = game_row.iloc[0].get("opponent_pitcher_hand", "") if not game_row.empty else ""
        row["opponent_pitcher_name"] = game_row.iloc[0].get("opponent_pitcher_name", "") if not game_row.empty else ""
        row.update(_handedness_features(latest, row["pitcher_throws_left"]))

        _add_recent_form(row, player_data, latest)

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
                row["pitcher_era"] = _safe_float(game_info.get("pitcher_era", 4.50), 4.50)
                row["pitcher_whip"] = _safe_float(game_info.get("pitcher_whip", 1.30), 1.30)
                row["park_factor"] = _park_factor_for_game(game_info)
                row["hitter_park_boost"] = 1 if row["park_factor"] > 1.02 else 0

        row.setdefault("away_team", "")
        row.setdefault("home_team", "")
        row.setdefault("game_datetime", "")
        row.setdefault("pitcher_era", 4.50)
        row.setdefault("pitcher_whip", 1.30)
        row.setdefault("park_factor", 1.0)
        row.setdefault("hitter_park_boost", 0)
        candidates.append(row)

    df = pd.DataFrame(candidates)
    if df.empty:
        return df

    # Once lineups are confirmed, avoid bench players and low-PA 8/9 hitters.
    confirmed = df["lineup_position"].fillna(0).astype(int) > 0
    if confirmed.any():
        df = df[(df["lineup_position"] >= MIN_LINEUP_POSITION) & (df["lineup_position"] <= MAX_LINEUP_POSITION)]
    return df

def predict_picks(model, feature_cols: list, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict hit probabilities for all candidates.
    """
    # Ensure we have the exact features the model expects
    for col in feature_cols:
        if col not in candidates.columns:
            candidates[col] = 0
    X = candidates[feature_cols].fillna(0)
    
    # Predict
    probs = model.predict_proba(X)[:, 1]
    candidates = candidates.reset_index(drop=True)
    candidates["p_hit"] = probs
    
    qualified = candidates[candidates["p_hit"] >= MIN_HIT_PROBABILITY].copy()
    if qualified.empty:
        logger.warning("No candidates cleared MIN_HIT_PROBABILITY=%.2f; returning top raw probabilities instead.", MIN_HIT_PROBABILITY)
        qualified = candidates.copy()

    # Sort by probability and select top N
    top_picks = qualified.sort_values(["p_hit", "player_name", "player_id"], ascending=[False, True, True]).head(TOP_N_PICKS)
    
    return top_picks, candidates


def save_candidates(candidates: pd.DataFrame, timestamp: str) -> None:
    os.makedirs(PICKS_DIR, exist_ok=True)
    out_path = os.path.join(PICKS_DIR, f"all_candidates_{timestamp}.csv")
    output_cols = [
        "player_id", "player_name", "position", "lineup_position",
        "batter_bat_side", "opponent_pitcher_name", "opponent_pitcher_hand", "platoon_advantage",
        "matchup_split_avg", "matchup_split_ops", "matchup_split_pa",
        "away_team", "home_team", "game_datetime", "p_hit"
    ]
    output_cols = [c for c in output_cols if c in candidates.columns]
    candidates.sort_values(["p_hit", "player_name", "player_id"], ascending=[False, True, True])[output_cols].to_csv(out_path, index=False)
    logger.info(f"Saved all candidates to {out_path}")


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
    picks, scored_candidates = predict_picks(model, feature_cols, candidates)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_candidates(scored_candidates, timestamp)
    
    # Display picks
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TOP {TOP_N_PICKS} PICKS FOR TODAY")
    logger.info(f"{'=' * 60}")
    
    for i, (_, row) in enumerate(picks.iterrows(), 1):
        logger.info(
            f"{i}. {row['player_name']} ({row['position']}, batting {int(row.get('lineup_position', 0)) or 'TBD'}) | "
            f"Game: {row['away_team']} @ {row['home_team']} | "
            f"Pitcher: {row.get('opponent_pitcher_name', '')} {row.get('opponent_pitcher_hand', '')} | "
            f"PlatoonAdv={int(row.get('platoon_advantage', 0))} | P(Hit) = {row['p_hit']:.3f}"
        )
    
    # Save picks
    os.makedirs(PICKS_DIR, exist_ok=True)
    out_path = os.path.join(PICKS_DIR, f"picks_{timestamp}.csv")
    
    output_cols = [
        "player_id", "player_name", "position", "lineup_position",
        "batter_bat_side", "opponent_pitcher_name", "opponent_pitcher_hand", "platoon_advantage",
        "matchup_split_avg", "matchup_split_ops", "matchup_split_pa",
        "away_team", "home_team", "game_datetime", "p_hit"
    ]
    output_cols = [c for c in output_cols if c in picks.columns]
    picks[output_cols].to_csv(out_path, index=False)
    logger.info(f"\nSaved picks to {out_path}")


if __name__ == "__main__":
    main()
