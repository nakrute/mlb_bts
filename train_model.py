import logging
import os
import warnings

import pandas as pd
import statsapi

import data_fetcher as df_get
from config import DATA_DIR, LOGS_DIR, MODEL_DIR, PICKS_DIR

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
today = pd.to_datetime("today").normalize().strftime("%Y-%m-%d")


def ensure_directories() -> None:
    for path in [DATA_DIR, os.path.join(DATA_DIR, "input"), MODEL_DIR, PICKS_DIR, LOGS_DIR]:
        os.makedirs(path, exist_ok=True)


def _first_present(*values):
    for value in values:
        if value not in (None, "", "--"):
            return value
    return None


def _normalize_hand(value):
    value = str(value or "").strip().upper()
    if value.startswith("L"):
        return "L"
    if value.startswith("R"):
        return "R"
    if value.startswith("S"):
        return "S"
    return ""


def _hand_flags(batter_side, pitcher_hand):
    batter_side = _normalize_hand(batter_side)
    pitcher_hand = _normalize_hand(pitcher_hand)
    batter_switch = int(batter_side == "S")
    batter_bats_left = int(batter_side in {"L", "S"})
    batter_bats_right = int(batter_side in {"R", "S"})
    pitcher_throws_left = int(pitcher_hand == "L")
    platoon_advantage = int(
        batter_switch
        or (batter_side == "R" and pitcher_hand == "L")
        or (batter_side == "L" and pitcher_hand == "R")
    )
    return batter_bats_left, batter_bats_right, batter_switch, pitcher_throws_left, platoon_advantage


def _nested_code(blob, key):
    value = blob.get(key) if isinstance(blob, dict) else None
    if isinstance(value, dict):
        return _first_present(value.get("code"), value.get("abbreviation"), value.get("description"))
    return value


def _extract_player_context(player_blob):
    person = player_blob.get("person", {}) if isinstance(player_blob, dict) else {}
    position = player_blob.get("position", {}) if isinstance(player_blob, dict) else {}
    bat_side = _first_present(_nested_code(player_blob, "batSide"), _nested_code(person, "batSide"))
    pitch_hand = _first_present(_nested_code(player_blob, "pitchHand"), _nested_code(person, "pitchHand"))
    batting_order = _first_present(player_blob.get("battingOrder"), player_blob.get("batting_order"))
    try:
        lineup_position = int(str(batting_order)[:1]) if batting_order not in (None, "") else 0
    except Exception:
        lineup_position = 0
    pos = position.get("abbreviation") if isinstance(position, dict) else position
    return {
        "player_name": _first_present(person.get("fullName"), player_blob.get("fullName"), player_blob.get("name")),
        "position": _first_present(pos, player_blob.get("position")),
        "batting_order_raw": batting_order or "",
        "lineup_position": lineup_position,
        "batter_bat_side": _normalize_hand(bat_side),
        "pitch_hand": _normalize_hand(pitch_hand),
    }


def enrich_todays_context(today_str: str) -> None:
    input_dir = os.path.join(DATA_DIR, "input")
    players_path = os.path.join(input_dir, f"{today_str}_players_games.csv")
    games_path = os.path.join(input_dir, f"{today_str}_todays_games.csv")
    if not os.path.exists(players_path) or not os.path.exists(games_path):
        logger.warning("Skipping lineup/platoon enrichment because input CSVs are missing.")
        return

    players_games = pd.read_csv(players_path)
    todays_games = pd.read_csv(games_path)
    if players_games.empty or todays_games.empty:
        logger.warning("Skipping lineup/platoon enrichment because input CSVs are empty.")
        return

    enrich_rows = []
    game_pitcher_hands = {}
    for _, game in todays_games.iterrows():
        game_id = game.get("game_id")
        try:
            box = statsapi.boxscore_data(int(game_id))
        except Exception as exc:
            logger.debug("Could not enrich game %s: %s", game_id, exc)
            continue

        side_context = {}
        for side in ["home", "away"]:
            players = box.get(side, {}).get("players", {}) if isinstance(box, dict) else {}
            probable_name = game.get(f"{side}_probable_pitcher", "")
            probable_hand = ""
            for _, player_blob in players.items():
                ctx = _extract_player_context(player_blob)
                if ctx.get("player_name") and str(ctx["player_name"]).lower() == str(probable_name).lower():
                    probable_hand = ctx.get("pitch_hand", "")
                    break
            side_context[side] = {"probable_pitcher": probable_name, "probable_hand": probable_hand}
            game_pitcher_hands[(game_id, side)] = probable_hand

        for side in ["home", "away"]:
            opponent = "away" if side == "home" else "home"
            players = box.get(side, {}).get("players", {}) if isinstance(box, dict) else {}
            for key, player_blob in players.items():
                player_id = str(key).replace("ID", "")
                if not player_id.isdigit():
                    continue
                ctx = _extract_player_context(player_blob)
                opponent_hand = side_context.get(opponent, {}).get("probable_hand", "")
                b_left, b_right, b_switch, p_left, platoon = _hand_flags(ctx.get("batter_bat_side"), opponent_hand)
                enrich_rows.append({
                    "player_id": int(player_id),
                    "game_id": game_id,
                    "team_side": side,
                    "team_name": game.get(f"{side}_name", ""),
                    "opponent_pitcher_name": side_context.get(opponent, {}).get("probable_pitcher", ""),
                    "opponent_pitcher_hand": opponent_hand,
                    "batter_bat_side": ctx.get("batter_bat_side", ""),
                    "lineup_position": ctx.get("lineup_position", 0),
                    "batting_order_raw": ctx.get("batting_order_raw", ""),
                    "batter_bats_left": b_left,
                    "batter_bats_right": b_right,
                    "batter_switch_hitter": b_switch,
                    "pitcher_throws_left": p_left,
                    "platoon_advantage": platoon,
                })

    if enrich_rows:
        enrich_df = pd.DataFrame(enrich_rows).drop_duplicates(["player_id", "game_id"], keep="last")
        merged = players_games.merge(enrich_df, on=["player_id", "game_id"], how="left")
        defaults = {
            "team_side": "", "team_name": "", "opponent_pitcher_name": "",
            "opponent_pitcher_hand": "", "batter_bat_side": "", "batting_order_raw": "",
            "lineup_position": 0, "batter_bats_left": 0, "batter_bats_right": 0,
            "batter_switch_hitter": 0, "pitcher_throws_left": 0, "platoon_advantage": 0,
        }
        for col, default in defaults.items():
            if col not in merged.columns:
                merged[col] = default
            merged[col] = merged[col].fillna(default)
        merged.to_csv(players_path, index=False)
        logger.info("Enriched %s player-game rows with lineup and platoon context", len(enrich_df))
    else:
        logger.warning("No lineup/platoon context found. This often happens before confirmed lineups are posted.")

    for side in ["home", "away"]:
        col = f"{side}_probable_pitcher_hand"
        if col not in todays_games.columns:
            todays_games[col] = ""
        todays_games[col] = todays_games.apply(lambda r: game_pitcher_hands.get((r.get("game_id"), side), r.get(col, "")), axis=1)
    todays_games.to_csv(games_path, index=False)


if __name__ == "__main__":
    ensure_directories()
    logger.info("=" * 60)
    logger.info("BTS MODEL — INPUT DATA PIPELINE")
    logger.info("=" * 60)
    df = df_get.historical_data_by_today_players(today)
    pitchers = df_get.get_historical_pitching_logs(df, today=today)
    enrich_todays_context(today)
    logger.info("Fetched %s batter rows and %s pitcher rows", len(df), len(pitchers))
    logger.info("Input data complete. Next run: python build_training_from_input.py")
