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

from config import (
    DATA_DIR, MODEL_DIR, PICKS_DIR, TOP_N_PICKS, PARK_FACTORS, TEAM_ABBREVS,
    MIN_SEASON_PA, MIN_HIT_PROBABILITY, MIN_LINEUP_POSITION, MAX_LINEUP_POSITION,
    CONFIDENCE_THRESHOLD,
)

# Optional smart-selector config. Defaults keep backward compatibility if config.py was not updated.
try:
    from config import (
        ALLOW_SKIP_DAYS,
        MIN_ADJUSTED_HIT_PROBABILITY,
        GREEN_LIGHT_THRESHOLD,
        BORDERLINE_THRESHOLD,
        MAX_PICKS_PER_GAME,
        ELITE_PITCHER_ERA,
        ELITE_PITCHER_WHIP,
        LATE_LINEUP_PENALTY_START,
        RISK_PENALTIES,
    )
except ImportError:
    ALLOW_SKIP_DAYS = True
    MIN_ADJUSTED_HIT_PROBABILITY = MIN_HIT_PROBABILITY
    GREEN_LIGHT_THRESHOLD = CONFIDENCE_THRESHOLD
    BORDERLINE_THRESHOLD = MIN_HIT_PROBABILITY
    MAX_PICKS_PER_GAME = 1
    ELITE_PITCHER_ERA = 3.30
    ELITE_PITCHER_WHIP = 1.15
    LATE_LINEUP_PENALTY_START = 6
    RISK_PENALTIES = {
        "elite_pitcher": 0.045,
        "late_lineup": 0.025,
        "unconfirmed_lineup": 0.020,
        "no_platoon_advantage": 0.010,
        "low_recent_form": 0.020,
        "pitcher_park": 0.015,
    }

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


def build_candidate_rows(historical: pd.DataFrame, todays_games: pd.DataFrame,
                        players_games: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature rows for all players in today's games.
    Filters out pitchers and low-sample candidates.
    """
    candidates = []
    batters = historical[historical["position"] != "P"].copy()

    today_player_ids = players_games["player_id"].unique()

    for player_id in today_player_ids:
        player_data = batters[batters["id"] == player_id].sort_values("season", ascending=False)

        if player_data.empty:
            continue

        latest = player_data.iloc[0]
        player_pos = latest.get("position", "DH")
        season_pa = _safe_int(latest.get("atBats", 0), 0)

        if season_pa < MIN_SEASON_PA:
            continue

        row = {
            "player_id": player_id,
            "player_name": latest["name"],
            "position": player_pos,
            "current_avg": _safe_float(latest.get("avg", 0.270), 0.270),
            "current_hits": _safe_int(latest.get("hits", 0), 0),
            "current_at_bats": season_pa,
            "season_avg": _safe_float(latest.get("avg", 0.270), 0.270),
            "season_obp": _safe_float(latest.get("obp", 0.320), 0.320),
            "season_ops": _safe_float(latest.get("ops", 0.720), 0.720),
            "season_pa": season_pa,
            "season_hits": _safe_int(latest.get("hits", 0), 0),
            "last_avg": _safe_float(latest.get("avg", 0.270), 0.270),
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

def _risk_flags_and_penalty(row: pd.Series) -> tuple[list[str], float]:
    """Apply conservative BTS risk penalties after the model score.

    p_hit remains the raw calibrated model probability. adjusted_p_hit is used
    for final streak-survival decisions.
    """
    flags = []
    penalty = 0.0

    pitcher_era = _safe_float(row.get("pitcher_era", 4.50), 4.50)
    pitcher_whip = _safe_float(row.get("pitcher_whip", 1.30), 1.30)
    lineup_position = _safe_int(row.get("lineup_position", 0), 0)
    roll7_hit_rate = _safe_float(row.get("roll7_hit_game_rate", 0.0), 0.0)
    park_factor = _safe_float(row.get("park_factor", 1.0), 1.0)
    platoon_advantage = _safe_int(row.get("platoon_advantage", 0), 0)

    if pitcher_era <= ELITE_PITCHER_ERA and pitcher_whip <= ELITE_PITCHER_WHIP:
        flags.append("elite_pitcher")
        penalty += RISK_PENALTIES.get("elite_pitcher", 0.0)

    if lineup_position == 0:
        flags.append("unconfirmed_lineup")
        penalty += RISK_PENALTIES.get("unconfirmed_lineup", 0.0)
    elif lineup_position >= LATE_LINEUP_PENALTY_START:
        flags.append("late_lineup")
        penalty += RISK_PENALTIES.get("late_lineup", 0.0)

    if platoon_advantage == 0:
        flags.append("no_platoon_advantage")
        penalty += RISK_PENALTIES.get("no_platoon_advantage", 0.0)

    if roll7_hit_rate and roll7_hit_rate < 0.50:
        flags.append("low_recent_form")
        penalty += RISK_PENALTIES.get("low_recent_form", 0.0)

    if park_factor < 0.98:
        flags.append("pitcher_park")
        penalty += RISK_PENALTIES.get("pitcher_park", 0.0)

    return flags, min(penalty, 0.15)


def _confidence_tier(adjusted_p_hit: float) -> str:
    if adjusted_p_hit >= GREEN_LIGHT_THRESHOLD:
        return "GREEN"
    if adjusted_p_hit >= BORDERLINE_THRESHOLD:
        return "BORDERLINE"
    return "PASS"


def _select_diversified_picks(qualified: pd.DataFrame) -> pd.DataFrame:
    """Select top picks while avoiding too much same-game correlation."""
    if qualified.empty:
        return qualified

    selected_rows = []
    game_counts = {}

    for _, row in qualified.sort_values("adjusted_p_hit", ascending=False).iterrows():
        game_id = row.get("game_id", "")
        current_game_count = game_counts.get(game_id, 0)

        if game_id and current_game_count >= MAX_PICKS_PER_GAME:
            continue

        selected_rows.append(row)
        if game_id:
            game_counts[game_id] = current_game_count + 1

        if len(selected_rows) >= TOP_N_PICKS:
            break

    if len(selected_rows) < min(TOP_N_PICKS, len(qualified)):
        selected_ids = {r.get("player_id") for r in selected_rows}
        for _, row in qualified.sort_values("adjusted_p_hit", ascending=False).iterrows():
            if row.get("player_id") in selected_ids:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= TOP_N_PICKS:
                break

    return pd.DataFrame(selected_rows)


def predict_picks(model, feature_cols: list, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Predict hit probabilities, apply BTS risk controls, and choose only qualified picks.
    """
    for col in feature_cols:
        if col not in candidates.columns:
            candidates[col] = 0

    X = candidates[feature_cols].fillna(0)

    probs = model.predict_proba(X)[:, 1]
    candidates = candidates.reset_index(drop=True)
    candidates["p_hit"] = probs

    penalties = candidates.apply(_risk_flags_and_penalty, axis=1)
    candidates["risk_flags"] = penalties.map(lambda x: ",".join(x[0]) if x[0] else "")
    candidates["risk_penalty"] = penalties.map(lambda x: x[1])
    candidates["adjusted_p_hit"] = (candidates["p_hit"] - candidates["risk_penalty"]).clip(lower=0, upper=1)
    candidates["confidence_tier"] = candidates["adjusted_p_hit"].map(_confidence_tier)

    qualified = candidates[
        (candidates["p_hit"] >= MIN_HIT_PROBABILITY)
        & (candidates["adjusted_p_hit"] >= MIN_ADJUSTED_HIT_PROBABILITY)
    ].copy()

    if qualified.empty:
        best = candidates.sort_values("adjusted_p_hit", ascending=False).head(5)
        logger.warning(
            "No picks cleared smart selector thresholds "
            "(raw >= %.2f and adjusted >= %.2f). Best adjusted candidates: %s",
            MIN_HIT_PROBABILITY,
            MIN_ADJUSTED_HIT_PROBABILITY,
            best[["player_name", "p_hit", "adjusted_p_hit", "risk_flags"]].to_dict("records")
        )
        if ALLOW_SKIP_DAYS:
            return candidates.iloc[0:0].copy()
        qualified = candidates.copy()

    return _select_diversified_picks(qualified)

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
    logger.info(f"SMART SELECTOR PICKS FOR TODAY (max {TOP_N_PICKS})")
    logger.info(f"{'=' * 60}")
    
    if picks.empty:
        logger.info("PASS DAY: no candidate cleared the smart selector thresholds.")
    for i, (_, row) in enumerate(picks.iterrows(), 1):
        logger.info(
            f"{i}. {row['player_name']} ({row['position']}, batting {int(row.get('lineup_position', 0)) or 'TBD'}) | "
            f"Game: {row['away_team']} @ {row['home_team']} | "
            f"Pitcher: {row.get('opponent_pitcher_name', '')} {row.get('opponent_pitcher_hand', '')} | "
            f"PlatoonAdv={int(row.get('platoon_advantage', 0))} | "
            f"P(raw)={row['p_hit']:.3f} | P(adj)={row['adjusted_p_hit']:.3f} | "
            f"Tier={row['confidence_tier']} | Risk={row.get('risk_flags', '') or 'none'}"
        )
    
    # Save picks
    os.makedirs(PICKS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(PICKS_DIR, f"picks_{timestamp}.csv")
    
    output_cols = [
        "player_id", "player_name", "position", "lineup_position",
        "batter_bat_side", "opponent_pitcher_name", "opponent_pitcher_hand", "platoon_advantage",
        "away_team", "home_team", "game_datetime",
        "p_hit", "risk_penalty", "adjusted_p_hit", "confidence_tier", "risk_flags"
    ]
    output_cols = [c for c in output_cols if c in picks.columns]
    picks[output_cols].to_csv(out_path, index=False)
    logger.info(f"\nSaved picks to {out_path}")


if __name__ == "__main__":
    main()
