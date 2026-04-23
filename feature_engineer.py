# feature_engineer.py — Builds the ML feature vector for each player-game
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    PARK_FACTORS, ROLLING_WINDOWS, MIN_AB_FOR_PLATOON, MIN_AB_H2H
)
from data_fetcher import (
    get_batter_season_stats, get_batter_game_log, get_batter_vs_pitcher,
    get_platoon_splits, get_batter_career_stats,
    get_pitcher_season_stats, get_pitcher_recent_form,
    get_weather, _safe_div
)

logger = logging.getLogger(__name__)


def build_feature_row(
    player_id: int,
    player_name: str,
    lineup_pos: int,
    bats: str,
    pitcher_id: Optional[int],
    pitcher_hand: str,
    home_or_away: str,
    park_abbrev: str,
    venue_name: str,
    weather_api_key: str = "",
) -> dict:
    """
    Builds a complete feature dict for one batter in one game.
    This is the core function that assembles all signals.
    """
    features = {
        "player_id":   player_id,
        "player_name": player_name,
    }

    # ── 1. Season Stats ────────────────────────────────────────────────────────
    season = get_batter_season_stats(player_id)
    features.update(season)

    # ── 2. Career Stats (for regression-to-mean) ──────────────────────────────
    career = get_batter_career_stats(player_id)
    features.update(career)

    # ── 3. Rolling Window Features ────────────────────────────────────────────
    game_log = get_batter_game_log(player_id, num_days=max(ROLLING_WINDOWS) + 5)
    for window in ROLLING_WINDOWS:
        recent = game_log.tail(window) if not game_log.empty else pd.DataFrame()
        if not recent.empty and recent["ab"].sum() > 0:
            total_ab   = recent["ab"].sum()
            total_hits = recent["hits"].sum()
            games_with_hit = recent["got_hit"].sum()
            total_games    = len(recent)
            features[f"roll{window}_avg"]           = _safe_div(total_hits, total_ab)
            features[f"roll{window}_hit_game_rate"] = _safe_div(games_with_hit, total_games)
            features[f"roll{window}_ab_per_game"]   = _safe_div(total_ab, total_games)
            features[f"roll{window}_k_pct"]         = _safe_div(recent["so"].sum(), total_ab)
            features[f"roll{window}_bb_pct"]        = _safe_div(recent["bb"].sum(), total_ab)
        else:
            features[f"roll{window}_avg"]           = features.get("season_avg", 0.250)
            features[f"roll{window}_hit_game_rate"] = 0.65
            features[f"roll{window}_ab_per_game"]   = 3.5
            features[f"roll{window}_k_pct"]         = features.get("season_so_pct", 0.20)
            features[f"roll{window}_bb_pct"]        = features.get("season_bb_pct", 0.08)

    # ── 4. Streak Features ────────────────────────────────────────────────────
    if not game_log.empty:
        features["current_streak"]    = _compute_streak(game_log)
        features["last5_hit_games"]   = int(game_log.tail(5)["got_hit"].sum())
        features["last10_hit_games"]  = int(game_log.tail(10)["got_hit"].sum())
    else:
        features["current_streak"]   = 0
        features["last5_hit_games"]  = 3
        features["last10_hit_games"] = 7

    # ── 5. Platoon Splits ─────────────────────────────────────────────────────
    splits = get_platoon_splits(player_id)
    features.update(splits)

    # Effective AVG given platoon matchup
    if pitcher_hand == "L" and splits.get("vs_lhp_ab", 0) >= MIN_AB_FOR_PLATOON:
        features["platoon_avg"] = splits["vs_lhp_avg"]
        features["platoon_ops"] = splits["vs_lhp_ops"]
    elif pitcher_hand == "R" and splits.get("vs_rhp_ab", 0) >= MIN_AB_FOR_PLATOON:
        features["platoon_avg"] = splits["vs_rhp_avg"]
        features["platoon_ops"] = splits["vs_rhp_ops"]
    else:
        features["platoon_avg"] = features.get("season_avg", 0.250)
        features["platoon_ops"] = features.get("season_ops", 0.700)

    # Same-handed pitcher = harder (platoon disadvantage)
    features["platoon_disadvantage"] = int(
        (bats == "L" and pitcher_hand == "L") or
        (bats == "R" and pitcher_hand == "R")
    )

    # ── 6. Head-to-Head ───────────────────────────────────────────────────────
    if pitcher_id:
        h2h = get_batter_vs_pitcher(player_id, pitcher_id)
        features.update(h2h)
        features["h2h_sufficient"] = int(h2h["h2h_ab"] >= MIN_AB_H2H)
    else:
        features.update({"h2h_ab": 0, "h2h_hits": 0, "h2h_avg": 0.0, "h2h_so": 0})
        features["h2h_sufficient"] = 0

    # ── 7. Pitcher Features ───────────────────────────────────────────────────
    pitcher_stats = get_pitcher_season_stats(pitcher_id)
    features.update(pitcher_stats)

    pitcher_recent = get_pitcher_recent_form(pitcher_id)
    features.update(pitcher_recent)

    # ── 8. Ballpark ───────────────────────────────────────────────────────────
    features["park_factor"]   = PARK_FACTORS.get(park_abbrev, 100) / 100.0
    features["is_home"]       = 1 if home_or_away == "home" else 0
    features["lineup_pos"]    = lineup_pos

    # ── 9. Weather ────────────────────────────────────────────────────────────
    weather = get_weather(venue_name, weather_api_key)
    features["temp_f"]   = weather["temp_f"]
    features["wind_mph"] = weather["wind_mph"]
    features["is_dome"]  = weather["is_dome"]

    # Wind blowing out (180-270 deg) boosts offense
    wd = weather["wind_dir"]
    features["wind_out"] = int(150 <= wd <= 270 and weather["wind_mph"] > 8 and not weather["is_dome"])

    # Cold weather suppresses offense
    features["cold_game"] = int(weather["temp_f"] < 50 and not weather["is_dome"])

    # ── 10. Composite / Interaction Features ──────────────────────────────────
    # Weighted average of season avg + recent form (recency-weighted)
    features["composite_avg"] = (
        0.25 * features.get("season_avg", 0.250) +
        0.35 * features.get("roll14_avg", 0.250) +
        0.40 * features.get("roll7_avg",  0.250)
    )

    # Pitcher quality score (higher = tougher pitcher)
    features["pitcher_quality"] = (
        0.4 * _normalize(features.get("pitcher_era", 4.0), low=2.5, high=6.0) +
        0.3 * _normalize(features.get("pitcher_whip", 1.3), low=0.9, high=1.8) +
        0.3 * _normalize(features.get("pitcher_k9", 8.0), low=4.0, high=14.0)
    )

    # Expected PA (lineup position proxy — higher lineup pos = more PA)
    features["expected_pa"] = max(0, 4.5 - (lineup_pos - 1) * 0.15)

    return features


# ──────────────────────────────────────────────────────────────────────────────
# Build full DataFrame of today's candidates
# ──────────────────────────────────────────────────────────────────────────────

def build_daily_features(games: list[dict], weather_api_key: str = "") -> pd.DataFrame:
    """
    Takes today's game list, fetches all lineups and pitchers,
    builds and returns the full feature DataFrame (one row per batter).
    """
    from data_fetcher import get_lineup, get_starting_pitcher

    rows = []
    for game in games:
        game_id = game["game_id"]
        logger.info(f"Processing: {game['away_team']} @ {game['home_team']}")

        for side, opp_side in [("home", "away"), ("away", "home")]:
            lineup     = get_lineup(game_id, side)
            pitcher    = get_starting_pitcher(game_id, opp_side)  # pitcher the lineup faces
            park_abbrev = game["home_abbrev"]

            if not lineup:
                logger.warning(f"No lineup found for {side} team in game {game_id}")
                continue

            for player in lineup:
                try:
                    row = build_feature_row(
                        player_id      = player["player_id"],
                        player_name    = player["player_name"],
                        lineup_pos     = player["lineup_position"],
                        bats           = player["bats"],
                        pitcher_id     = pitcher["pitcher_id"],
                        pitcher_hand   = pitcher["hand"],
                        home_or_away   = side,
                        park_abbrev    = park_abbrev,
                        venue_name     = game["venue"],
                        weather_api_key = weather_api_key,
                    )
                    row["game_id"]      = game_id
                    row["opponent"]     = game[f"{opp_side}_team"]
                    row["pitcher_name"] = pitcher["pitcher_name"]
                    rows.append(row)
                except Exception as e:
                    logger.error(f"Feature build error for {player.get('player_name')}: {e}")

    df = pd.DataFrame(rows)
    logger.info(f"Built features for {len(df)} players")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_streak(game_log: pd.DataFrame) -> int:
    """Count current consecutive games with a hit."""
    streak = 0
    for got_hit in reversed(game_log["got_hit"].tolist()):
        if got_hit == 1:
            streak += 1
        else:
            break
    return streak


def _normalize(val: float, low: float, high: float) -> float:
    """Normalize value to [0, 1] range."""
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (val - low) / (high - low)))


def get_feature_columns() -> list[str]:
    """Returns the ordered list of feature columns used in training/inference."""
    cols = []
    # Season
    cols += ["season_pa", "season_ab", "season_hits", "season_avg", "season_obp",
             "season_slg", "season_ops", "season_so_pct", "season_bb_pct", "season_games"]
    # Career
    cols += ["career_games", "career_avg", "career_obp", "career_slg"]
    # Rolling windows
    for w in ROLLING_WINDOWS:
        cols += [f"roll{w}_avg", f"roll{w}_hit_game_rate", f"roll{w}_ab_per_game",
                 f"roll{w}_k_pct", f"roll{w}_bb_pct"]
    # Streak
    cols += ["current_streak", "last5_hit_games", "last10_hit_games"]
    # Platoon
    cols += ["vs_lhp_avg", "vs_lhp_ops", "vs_lhp_ab",
             "vs_rhp_avg", "vs_rhp_ops", "vs_rhp_ab",
             "platoon_avg", "platoon_ops", "platoon_disadvantage"]
    # H2H
    cols += ["h2h_ab", "h2h_hits", "h2h_avg", "h2h_so", "h2h_sufficient"]
    # Pitcher
    cols += ["pitcher_era", "pitcher_whip", "pitcher_baa", "pitcher_k9",
             "pitcher_bb9", "pitcher_hr9", "pitcher_ip", "pitcher_games",
             "recent_era", "recent_whip", "recent_baa"]
    # Context
    cols += ["park_factor", "is_home", "lineup_pos",
             "temp_f", "wind_mph", "is_dome", "wind_out", "cold_game"]
    # Composite
    cols += ["composite_avg", "pitcher_quality", "expected_pa"]
    return cols