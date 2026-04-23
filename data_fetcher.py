# data_fetcher.py — Pulls all raw data from MLB Stats API & pybaseball
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import statsapi

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Schedule & Lineup
# ──────────────────────────────────────────────────────────────────────────────

def get_todays_games(date: Optional[str] = None) -> list[dict]:
    """
    Returns list of today's scheduled games with home/away teams and game IDs.
    date format: 'YYYY-MM-DD', defaults to today.
    """
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    try:
        schedule = statsapi.schedule(date=date)
        games = []
        for g in schedule:
            games.append({
                "game_id":       g["game_id"],
                "game_datetime": g.get("game_datetime", ""),
                "status":        g.get("status", ""),
                "home_team":     g["home_name"],
                "home_id":       g["home_id"],
                "away_team":     g["away_name"],
                "away_id":       g["away_id"],
                "venue":         g.get("venue_name", ""),
                "home_abbrev":   _team_name_to_abbrev(g["home_name"]),
                "away_abbrev":   _team_name_to_abbrev(g["away_name"]),
            })
        logger.info(f"Found {len(games)} games on {date}")
        return games
    except Exception as e:
        logger.error(f"Error fetching schedule: {e}")
        return []


def get_starting_pitcher(game_id: int, home_or_away: str) -> dict:
    """
    Fetches the probable starting pitcher for a given game side.
    Returns dict with pitcher_id, name, hand.
    """
    try:
        boxscore = statsapi.boxscore_data(game_id)
        side = "home" if home_or_away == "home" else "away"
        pitcher_id = boxscore[side].get("probablePitcherId")
        if pitcher_id:
            info = statsapi.player_stat_data(pitcher_id, group="pitching", type="season")
            return {
                "pitcher_id":   pitcher_id,
                "pitcher_name": info.get("full_name", "Unknown"),
                "hand":         info.get("pitchHand", {}).get("code", "R"),
            }
    except Exception as e:
        logger.warning(f"Could not fetch pitcher for game {game_id}: {e}")
    return {"pitcher_id": None, "pitcher_name": "Unknown", "hand": "R"}


def get_lineup(game_id: int, home_or_away: str) -> list[dict]:
    """
    Returns confirmed lineup for a game side as a list of player dicts.
    Each dict has: player_id, name, lineup_position, bats (L/R/S).
    Falls back to roster if lineup not yet posted.
    """
    try:
        boxscore = statsapi.boxscore_data(game_id)
        side = "home" if home_or_away == "home" else "away"
        batters = boxscore[side].get("battingOrder", [])
        players = boxscore[side].get("players", {})

        lineup = []
        for pos, pid in enumerate(batters, start=1):
            key = f"ID{pid}"
            p = players.get(key, {})
            info = p.get("person", {})
            lineup.append({
                "player_id":      pid,
                "player_name":    info.get("fullName", "Unknown"),
                "lineup_position": pos,
                "bats":           p.get("batSide", {}).get("code", "R"),
            })
        return lineup
    except Exception as e:
        logger.warning(f"Could not fetch lineup for game {game_id} ({home_or_away}): {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Batter Stats
# ──────────────────────────────────────────────────────────────────────────────

def get_batter_season_stats(player_id: int, season: Optional[int] = None) -> dict:
    """Current season cumulative batting stats for a player."""
    if season is None:
        season = datetime.today().year
    try:
        stats = statsapi.player_stat_data(
            player_id, group="hitting", type="season", sportId=1
        )
        s = stats.get("stats", [{}])[0].get("stats", {})
        return {
            "season_pa":      int(s.get("plateAppearances", 0)),
            "season_ab":      int(s.get("atBats", 0)),
            "season_hits":    int(s.get("hits", 0)),
            "season_avg":     float(s.get("avg", 0) or 0),
            "season_obp":     float(s.get("obp", 0) or 0),
            "season_slg":     float(s.get("slg", 0) or 0),
            "season_ops":     float(s.get("ops", 0) or 0),
            "season_so_pct":  _safe_div(int(s.get("strikeOuts", 0)), int(s.get("plateAppearances", 1))),
            "season_bb_pct":  _safe_div(int(s.get("baseOnBalls", 0)), int(s.get("plateAppearances", 1))),
            "season_games":   int(s.get("gamesPlayed", 0)),
        }
    except Exception as e:
        logger.warning(f"Season stats error for {player_id}: {e}")
        return _empty_batter_stats("season")


def get_batter_game_log(player_id: int, num_days: int = 30) -> pd.DataFrame:
    """
    Returns a per-game log for the last `num_days` days.
    Columns: date, ab, hits, so, bb, hr.
    """
    end = datetime.today()
    start = end - timedelta(days=num_days + 5)  # buffer for off days

    try:
        log = statsapi.player_stat_data(
            player_id,
            group="hitting",
            type="gameLog",
            startDate=start.strftime("%Y-%m-%d"),
            endDate=end.strftime("%Y-%m-%d"),
        )
        rows = []
        for entry in log.get("stats", [{}])[0].get("splits", []):
            s = entry.get("stat", {})
            rows.append({
                "date":  entry.get("date", ""),
                "ab":    int(s.get("atBats", 0)),
                "hits":  int(s.get("hits", 0)),
                "so":    int(s.get("strikeOuts", 0)),
                "bb":    int(s.get("baseOnBalls", 0)),
                "hr":    int(s.get("homeRuns", 0)),
                "got_hit": 1 if int(s.get("hits", 0)) >= 1 else 0,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(num_days)
        return df
    except Exception as e:
        logger.warning(f"Game log error for {player_id}: {e}")
        return pd.DataFrame()


def get_batter_vs_pitcher(batter_id: int, pitcher_id: int) -> dict:
    """Career head-to-head stats: batter vs specific pitcher."""
    try:
        stats = statsapi.player_stat_data(
            batter_id,
            group="hitting",
            type="vsPlayer",
            opposingPlayerId=pitcher_id,
        )
        s = stats.get("stats", [{}])[0].get("stats", {})
        ab = int(s.get("atBats", 0))
        hits = int(s.get("hits", 0))
        return {
            "h2h_ab":    ab,
            "h2h_hits":  hits,
            "h2h_avg":   _safe_div(hits, ab),
            "h2h_so":    int(s.get("strikeOuts", 0)),
        }
    except Exception as e:
        logger.warning(f"H2H stats error ({batter_id} vs {pitcher_id}): {e}")
        return {"h2h_ab": 0, "h2h_hits": 0, "h2h_avg": 0.0, "h2h_so": 0}


def get_platoon_splits(player_id: int) -> dict:
    """Batter's splits vs LHP and RHP this season."""
    result = {}
    for split_type, key in [("vsLeft", "vs_lhp"), ("vsRight", "vs_rhp")]:
        try:
            stats = statsapi.player_stat_data(
                player_id, group="hitting", type=split_type
            )
            s = stats.get("stats", [{}])[0].get("stats", {})
            result[f"{key}_avg"] = float(s.get("avg", 0) or 0)
            result[f"{key}_ops"] = float(s.get("ops", 0) or 0)
            result[f"{key}_ab"]  = int(s.get("atBats", 0))
        except Exception:
            result[f"{key}_avg"] = 0.0
            result[f"{key}_ops"] = 0.0
            result[f"{key}_ab"]  = 0
    return result


def get_batter_career_stats(player_id: int) -> dict:
    """Career totals — used for sample size and streak context."""
    try:
        stats = statsapi.player_stat_data(
            player_id, group="hitting", type="career"
        )
        s = stats.get("stats", [{}])[0].get("stats", {})
        return {
            "career_games":  int(s.get("gamesPlayed", 0)),
            "career_avg":    float(s.get("avg", 0) or 0),
            "career_obp":    float(s.get("obp", 0) or 0),
            "career_slg":    float(s.get("slg", 0) or 0),
        }
    except Exception as e:
        logger.warning(f"Career stats error for {player_id}: {e}")
        return {"career_games": 0, "career_avg": 0.0, "career_obp": 0.0, "career_slg": 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# Pitcher Stats
# ──────────────────────────────────────────────────────────────────────────────

def get_pitcher_season_stats(pitcher_id: int) -> dict:
    """Current season stats for a starting pitcher."""
    if pitcher_id is None:
        return _empty_pitcher_stats()
    try:
        stats = statsapi.player_stat_data(
            pitcher_id, group="pitching", type="season"
        )
        s = stats.get("stats", [{}])[0].get("stats", {})
        ip = float(s.get("inningsPitched", 0) or 0)
        return {
            "pitcher_era":     float(s.get("era", 5.0) or 5.0),
            "pitcher_whip":    float(s.get("whip", 1.4) or 1.4),
            "pitcher_baa":     float(s.get("avg", 0.260) or 0.260),
            "pitcher_k9":      float(s.get("strikeoutsPer9Inn", 7.0) or 7.0),
            "pitcher_bb9":     float(s.get("walksPer9Inn", 3.0) or 3.0),
            "pitcher_hr9":     float(s.get("homeRunsPer9", 1.2) or 1.2),
            "pitcher_ip":      ip,
            "pitcher_games":   int(s.get("gamesStarted", 0)),
        }
    except Exception as e:
        logger.warning(f"Pitcher stats error for {pitcher_id}: {e}")
        return _empty_pitcher_stats()


def get_pitcher_recent_form(pitcher_id: int, num_starts: int = 3) -> dict:
    """ERA/WHIP/hits-allowed over last N starts."""
    if pitcher_id is None:
        return {"recent_era": 5.0, "recent_whip": 1.4, "recent_baa": 0.260}
    try:
        log = statsapi.player_stat_data(
            pitcher_id, group="pitching", type="gameLog"
        )
        splits = log.get("stats", [{}])[0].get("splits", [])
        # filter to starts only, take last N
        starts = [s for s in splits if int(s.get("stat", {}).get("gamesStarted", 0)) > 0]
        recent = starts[-num_starts:] if len(starts) >= num_starts else starts
        if not recent:
            return {"recent_era": 5.0, "recent_whip": 1.4, "recent_baa": 0.260}

        total_ip    = sum(float(s["stat"].get("inningsPitched", 0) or 0) for s in recent)
        total_er    = sum(int(s["stat"].get("earnedRuns", 0)) for s in recent)
        total_hits  = sum(int(s["stat"].get("hits", 0)) for s in recent)
        total_bb    = sum(int(s["stat"].get("baseOnBalls", 0)) for s in recent)
        total_ab    = sum(int(s["stat"].get("atBats", 0)) for s in recent)

        return {
            "recent_era":   (total_er / total_ip * 9) if total_ip > 0 else 5.0,
            "recent_whip":  ((total_hits + total_bb) / total_ip) if total_ip > 0 else 1.4,
            "recent_baa":   _safe_div(total_hits, total_ab) or 0.260,
        }
    except Exception as e:
        logger.warning(f"Pitcher recent form error for {pitcher_id}: {e}")
        return {"recent_era": 5.0, "recent_whip": 1.4, "recent_baa": 0.260}


def get_historical_game_logs(player_id: int, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Pull multi-season game logs for training.
    Returns DataFrame with one row per game-appearance.
    """
    all_rows = []
    try:
        log = statsapi.player_stat_data(
            player_id,
            group="hitting",
            type="yearByYear",
        )
        splits = log['stats']
        for entry in splits:
            all_rows.append({
                "id": log.get("id", ""),
                "first_name": log.get("first_name", ""),
                "last_name": log.get("last_name", ""),
                "season": entry.get("season", ""),
                "avg": entry['stats'].get("avg", 0),
                "atBats": entry['stats'].get("atBats", 0),
                "obp": entry['stats'].get("obp", 0),
                "slg": entry['stats'].get("slg", 0),
                "ops": entry['stats'].get("ops", 0),
                "homeRuns": entry['stats'].get("homeRuns", 0),
            })
    except Exception as e:
        logger.debug(f"Historical log error: player {player_id}: {e}")

    df = pd.DataFrame(all_rows)

    return df

BALLPARK_COORDS = {
    "Coors Field":                   (39.756, -104.994),
    "Fenway Park":                   (42.346, -71.097),
    "Wrigley Field":                 (41.948, -87.655),
    "Yankee Stadium":                (40.829, -73.926),
    "Dodger Stadium":                (34.074, -118.240),
    "Oracle Park":                   (37.778, -122.389),
    "Petco Park":                    (32.707, -117.157),
    "T-Mobile Park":                 (47.591, -122.332),
    "Tropicana Field":               (27.768, -82.653),
    "loanDepot park":                (25.778, -80.220),
    "American Family Field":         (43.028, -87.971),
    "Truist Park":                   (33.891, -84.468),
    "Globe Life Field":              (32.747, -97.083),
    "Minute Maid Park":              (29.757, -95.355),
    "Target Field":                  (44.982, -93.278),
    "PNC Park":                      (40.447, -80.006),
    "Camden Yards":                  (39.284, -76.622),
    "Citizens Bank Park":            (39.906, -75.166),
    "Great American Ball Park":      (39.097, -84.508),
}

def get_weather(venue_name: str, api_key: str) -> dict:
    """Fetch weather for a ballpark. Returns temp_f, wind_mph, wind_dir."""
    coords = BALLPARK_COORDS.get(venue_name)
    if not coords or not api_key:
        return {"temp_f": 72, "wind_mph": 5, "wind_dir": 0, "is_dome": _is_dome(venue_name)}

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={coords[0]}&lon={coords[1]}&appid={api_key}&units=imperial"
        )
        r = requests.get(url, timeout=5)
        data = r.json()
        return {
            "temp_f":   data["main"]["temp"],
            "wind_mph": data["wind"]["speed"],
            "wind_dir": data["wind"].get("deg", 0),
            "is_dome":  _is_dome(venue_name),
        }
    except Exception as e:
        logger.warning(f"Weather fetch failed for {venue_name}: {e}")
        return {"temp_f": 72, "wind_mph": 5, "wind_dir": 0, "is_dome": _is_dome(venue_name)}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

DOME_STADIUMS = {
    "Tropicana Field", "loanDepot park", "Rogers Centre",
    "Globe Life Field", "Minute Maid Park", "T-Mobile Park",  # retractable
    "American Family Field", "Chase Field",
}

def _is_dome(venue_name: str) -> int:
    return 1 if any(d.lower() in venue_name.lower() for d in DOME_STADIUMS) else 0


TEAM_ABBREVS = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}

def _team_name_to_abbrev(name: str) -> str:
    return TEAM_ABBREVS.get(name, name[:3].upper())


def _safe_div(a, b):
    try:
        return a / b if b and b != 0 else 0.0
    except Exception:
        return 0.0


def _empty_batter_stats(prefix: str) -> dict:
    return {
        f"{prefix}_pa": 0, f"{prefix}_ab": 0, f"{prefix}_hits": 0,
        f"{prefix}_avg": 0.0, f"{prefix}_obp": 0.0, f"{prefix}_slg": 0.0,
        f"{prefix}_ops": 0.0, f"{prefix}_so_pct": 0.0, f"{prefix}_bb_pct": 0.0,
        f"{prefix}_games": 0,
    }


def _empty_pitcher_stats() -> dict:
    return {
        "pitcher_era": 5.0, "pitcher_whip": 1.40, "pitcher_baa": 0.260,
        "pitcher_k9": 7.0, "pitcher_bb9": 3.5, "pitcher_hr9": 1.2,
        "pitcher_ip": 0, "pitcher_games": 0,
    }