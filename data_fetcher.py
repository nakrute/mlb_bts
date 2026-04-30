import logging
import time

import pandas as pd
import statsapi

from config import GAME_LOG_SEASONS_BACK

logger = logging.getLogger(__name__)
today = pd.to_datetime("today").normalize().strftime("%Y-%m-%d")


def get_todays_games(today: str=today) -> pd.DataFrame:
    games = statsapi.schedule()
    games_df = pd.DataFrame(games) 
    if not games_df.empty and "game_id" in games_df.columns:
        games_df = games_df.sort_values("game_id").reset_index(drop=True)
    games_df.to_csv(f"./data/input/{today}_todays_games.csv", index=False)
    return games_df


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
                        # stored as (player_id, game_id) for potential future use
                        player_ids.add((int(player_id_str), game['game_id']))
            time.sleep(0.02)  # Rate limiting
        except Exception as e:
            logger.debug(f"Error parsing boxscore for game {game['game_id']}: {e}")
            pass
    
    return player_ids


def historical_data_by_today_players(today: str=today) -> pd.DataFrame:
    today_games = get_todays_games(today)
    player_id_set = get_players_in_games(today_games)
    players_games_df = pd.DataFrame(sorted(player_id_set), columns=["player_id", "game_id"])
    players_games_df.to_csv(f"./data/input/{today}_players_games.csv", index=False)
    player_ids = sorted({x[0] for x in player_id_set})
    logger.info(f"Found {len(player_ids)} unique players in today's games")
    seasons = _game_log_seasons(today)
    
    all_df = pd.DataFrame()
    for count, pid in enumerate(player_ids, start=1):
        try:
            logging.info(f"Processing player {pid} ({count}/{len(player_ids)})")
            rows = get_historical_batting_game_logs(pid, seasons)
            splits = get_historical_batting_hand_splits(pid, seasons)
            if not rows.empty and not splits.empty:
                rows = rows.merge(splits, on=["id", "season"], how="left")
            if not rows.empty:
                all_df = pd.concat([all_df, rows], ignore_index=True)
        except Exception as e:
            logger.debug(f"Error fetching historical data for player {pid}: {e}")
            pass
    
    if not all_df.empty:
        all_df = all_df.sort_values(["id", "game_date", "game_id"]).reset_index(drop=True)
    all_df.to_csv(f"./data/input/{today}_historical_data.csv", index=False)
    logger.info(f"Built historical dataset with {len(all_df)} rows for today's players")
    return all_df


def _game_log_seasons(today_str: str) -> list[int]:
    current_year = pd.to_datetime(today_str).year
    return list(range(current_year - GAME_LOG_SEASONS_BACK, current_year + 1))


def _stat_value(stats: dict, key: str, default=0):
    value = stats.get(key, default)
    return default if value in (None, "", "--") else value


def get_historical_batting_game_logs(player_id: int, seasons: list[int]) -> pd.DataFrame:
    all_rows = []
    for season in seasons:
        try:
            data = statsapi.get("person", {
                "personId": player_id,
                "hydrate": f"stats(group=[hitting],type=gameLog,season={season},sportId=1),currentTeam",
            })
            people = data.get("people", [])
            if not people:
                continue
            person = people[0]
            player_name = f"{person.get('useName', '')} {person.get('lastName', '')}".strip()
            position = person.get("primaryPosition", {}).get("abbreviation", "")

            for stat_group in person.get("stats", []):
                for split in stat_group.get("splits", []):
                    stats = split.get("stat", {})
                    game = split.get("game", {})
                    all_rows.append({
                        "id": person.get("id", player_id),
                        "season": int(split.get("season", season)),
                        "game_date": split.get("date", ""),
                        "game_id": game.get("gamePk", ""),
                        "name": player_name,
                        "position": position,
                        "age": _stat_value(stats, "age", 0),
                        "gamesPlayed": _stat_value(stats, "gamesPlayed", 1),
                        "groundOuts": _stat_value(stats, "groundOuts", 0),
                        "airOuts": _stat_value(stats, "airOuts", 0),
                        "runs": _stat_value(stats, "runs", 0),
                        "doubles": _stat_value(stats, "doubles", 0),
                        "triples": _stat_value(stats, "triples", 0),
                        "homeRuns": _stat_value(stats, "homeRuns", 0),
                        "strikeOuts": _stat_value(stats, "strikeOuts", 0),
                        "baseOnBalls": _stat_value(stats, "baseOnBalls", 0),
                        "intentionalWalks": _stat_value(stats, "intentionalWalks", 0),
                        "hits": _stat_value(stats, "hits", 0),
                        "hitByPitch": _stat_value(stats, "hitByPitch", 0),
                        "avg": _stat_value(stats, "avg", 0),
                        "atBats": _stat_value(stats, "atBats", 0),
                        "obp": _stat_value(stats, "obp", 0),
                        "slg": _stat_value(stats, "slg", 0),
                        "ops": _stat_value(stats, "ops", 0),
                        "caughtStealing": _stat_value(stats, "caughtStealing", 0),
                        "stolenBases": _stat_value(stats, "stolenBases", 0),
                        "groundIntoDoublePlay": _stat_value(stats, "groundIntoDoublePlay", 0),
                        "numberOfPitches": _stat_value(stats, "numberOfPitches", 0),
                        "plateAppearances": _stat_value(stats, "plateAppearances", 0),
                        "totalBases": _stat_value(stats, "totalBases", 0),
                        "rbi": _stat_value(stats, "rbi", 0),
                        "leftOnBase": _stat_value(stats, "leftOnBase", 0),
                        "sacBunts": _stat_value(stats, "sacBunts", 0),
                        "sacFlies": _stat_value(stats, "sacFlies", 0),
                        "babip": _stat_value(stats, "babip", 0),
                        "groundOutsToAirouts": _stat_value(stats, "groundOutsToAirouts", 0),
                        "catchersInterference": _stat_value(stats, "catchersInterference", 0),
                        "atBatsPerHomeRun": _stat_value(stats, "atBatsPerHomeRun", 0),
                    })
            time.sleep(0.02)
        except Exception as e:
            logger.debug(f"Game log error: player {player_id}, season {season}: {e}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["id", "game_date", "game_id"]).reset_index(drop=True)
    return df


def _split_row(player_id: int, season: int, split_code: str, stats: dict) -> dict:
    prefix = "batter_vs_lhp" if split_code == "vl" else "batter_vs_rhp"
    at_bats = _stat_value(stats, "atBats", 0)
    hits = _stat_value(stats, "hits", 0)
    return {
        "id": player_id,
        "season": season,
        f"{prefix}_avg": _stat_value(stats, "avg", 0),
        f"{prefix}_obp": _stat_value(stats, "obp", 0),
        f"{prefix}_ops": _stat_value(stats, "ops", 0),
        f"{prefix}_hits": hits,
        f"{prefix}_at_bats": at_bats,
        f"{prefix}_pa": _stat_value(stats, "plateAppearances", at_bats),
        f"{prefix}_k_pct": _safe_rate(_stat_value(stats, "strikeOuts", 0), _stat_value(stats, "plateAppearances", at_bats)),
        f"{prefix}_bb_pct": _safe_rate(_stat_value(stats, "baseOnBalls", 0), _stat_value(stats, "plateAppearances", at_bats)),
    }


def _safe_rate(numerator, denominator) -> float:
    try:
        denominator = float(denominator)
        if denominator <= 0:
            return 0.0
        return float(numerator) / denominator
    except Exception:
        return 0.0


def get_historical_batting_hand_splits(player_id: int, seasons: list[int]) -> pd.DataFrame:
    rows_by_season = {}
    for season in seasons:
        rows_by_season[(player_id, season)] = {"id": player_id, "season": season}
        for split_code in ["vl", "vr"]:
            try:
                data = statsapi.get("person", {
                    "personId": player_id,
                    "hydrate": f"stats(group=[hitting],type=statSplits,season={season},sitCodes={split_code},sportId=1),currentTeam",
                })
                people = data.get("people", [])
                if not people:
                    continue
                for stat_group in people[0].get("stats", []):
                    for split in stat_group.get("splits", []):
                        rows_by_season[(player_id, season)].update(
                            _split_row(player_id, season, split_code, split.get("stat", {}))
                        )
                time.sleep(0.02)
            except Exception as e:
                logger.debug(f"Hand split error: player {player_id}, season {season}, split {split_code}: {e}")

    df = pd.DataFrame(rows_by_season.values())
    if not df.empty:
        df = df.sort_values(["id", "season"]).reset_index(drop=True)
    return df


def get_historical_batting_logs(player_id: int) -> pd.DataFrame:
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
                "season": entry.get("season", ""),
                "name": f"{log.get('first_name', '')} {log.get('last_name', '')}",
                "position": log.get("position", ""), 
                "age": entry['stats'].get("age", 0),
                "gamesPlayed": entry['stats'].get("gamesPlayed", 0),
                "groundOuts": entry['stats'].get("groundOuts", 0),
                "airOuts": entry['stats'].get("airOuts", 0),
                "runs": entry['stats'].get("runs", 0),
                "doubles": entry['stats'].get("doubles", 0),
                "triples": entry['stats'].get("triples", 0),
                "homeRuns": entry['stats'].get("homeRuns", 0),
                "strikeOuts": entry['stats'].get("strikeOuts", 0),
                "baseOnBalls": entry['stats'].get("baseOnBalls", 0),
                "intentionalWalks": entry['stats'].get("intentionalWalks", 0),
                "hits": entry['stats'].get("hits", 0),
                "hitByPitch": entry['stats'].get("hitByPitch", 0),
                "avg": entry['stats'].get("avg", 0),
                "atBats": entry['stats'].get("atBats", 0),
                "obp": entry['stats'].get("obp", 0),
                "slg": entry['stats'].get("slg", 0),
                "ops": entry['stats'].get("ops", 0),
                "caughtStealing": entry['stats'].get("caughtStealing", 0),
                "stolenBases": entry['stats'].get("stolenBases", 0),
                "stolenBasePercentage": entry['stats'].get("stolenBasePercentage", 0),
                "caughtStealingPercentage": entry['stats'].get("caughtStealingPercentage", 0),
                "groundIntoDoublePlay": entry['stats'].get("groundIntoDoublePlay", 0),
                "numberOfPitches": entry['stats'].get("numberOfPitches", 0),
                "plateAppearances": entry['stats'].get("plateAppearances", 0),
                "totalBases": entry['stats'].get("totalBases", 0),
                "rbi": entry['stats'].get("rbi", 0),
                "leftOnBase": entry['stats'].get("leftOnBase", 0),
                "sacBunts": entry['stats'].get("sacBunts", 0),
                "sacFlies": entry['stats'].get("sacFlies", 0),
                "babip": entry['stats'].get("babip", 0),
                "groundOutsToAirouts": entry['stats'].get("groundOutsToAirouts", 0),
                "catchersInterference": entry['stats'].get("catchersInterference", 0),
                "atBatsPerHomeRun": entry['stats'].get("atBatsPerHomeRun", 0)
            })
    except Exception as e:
        logger.debug(f"Historical log error: player {player_id}: {e}")

    df = pd.DataFrame(all_rows)

    return df


def get_historical_pitching_logs(player_df: pd.DataFrame, 
                                 today: str=today) -> pd.DataFrame:
    pitcher_df = player_df[player_df['position'] == "P"]
    pitcher_ids = sorted(pitcher_df['id'].dropna().unique())
    all_rows = []
    for count, player_id in enumerate(pitcher_ids, start=1):
        logging.info(f"Processing pitcher {player_id} ({count}/{len(pitcher_ids)})")
        try:
            log = statsapi.player_stat_data(
                player_id,
                group="pitching",
                type="yearByYear",
            )
            splits = log['stats']
            for entry in splits:
                all_rows.append({
                    "id": log.get("id", ""),
                    "name": f"{log.get('first_name', '')} {log.get('last_name', '')}",
                    "season": entry.get("season", ""),
                    'age': entry['stats'].get("age", 0),
                    'gamesPlayed': entry['stats'].get("gamesPlayed", 0),
                    'gamesStarted': entry['stats'].get("gamesStarted", 0),
                    'groundOuts': entry['stats'].get("groundOuts", 0),
                    'airOuts': entry['stats'].get("airOuts", 0),
                    'runs': entry['stats'].get("runs", 0),
                    'doubles': entry['stats'].get("doubles", 0),
                    'triples': entry['stats'].get("triples", 0),
                    'homeRuns': entry['stats'].get("homeRuns", 0),
                    'strikeOuts': entry['stats'].get("strikeOuts", 0),
                    'baseOnBalls': entry['stats'].get("baseOnBalls", 0),
                    'intentionalWalks': entry['stats'].get("intentionalWalks", 0),
                    'hits': entry['stats'].get("hits", 0),
                    'hitByPitch': entry['stats'].get("hitByPitch", 0),
                    'avg': entry['stats'].get("avg", 0),
                    'atBats': entry['stats'].get("atBats", 0),
                    'obp': entry['stats'].get("obp", 0),
                    'slg': entry['stats'].get("slg", 0),
                    'ops': entry['stats'].get("ops", 0),
                    'caughtStealing': entry['stats'].get("caughtStealing", 0),
                    'stolenBases': entry['stats'].get("stolenBases", 0),
                    'stolenBasePercentage': entry['stats'].get("stolenBasePercentage", 0),
                    'caughtStealingPercentage': entry['stats'].get("caughtStealingPercentage", 0),
                    'groundIntoDoublePlay': entry['stats'].get("groundIntoDoublePlay", 0),
                    'numberOfPitches': entry['stats'].get("numberOfPitches", 0),
                    'era': entry['stats'].get("era", 0),
                    'inningsPitched': entry['stats'].get("inningsPitched", 0),
                    'wins': entry['stats'].get("wins", 0),
                    'losses': entry['stats'].get("losses", 0),
                    'saves': entry['stats'].get("saves", 0),
                    'saveOpportunities': entry['stats'].get("saveOpportunities", 0),
                    'holds': entry['stats'].get("holds", 0),
                    'blownSaves': entry['stats'].get("blownSaves", 0),
                    'earnedRuns': entry['stats'].get("earnedRuns", 0),
                    'airOuts': entry['stats'].get("airOuts", 0),
                    'runs': entry['stats'].get("runs", 0),
                    'doubles': entry['stats'].get("doubles", 0),
                    'triples': entry['stats'].get("triples", 0),
                    'homeRuns': entry['stats'].get("homeRuns", 0),
                    'strikeOuts': entry['stats'].get("strikeOuts", 0),
                    'baseOnBalls': entry['stats'].get("baseOnBalls", 0),
                    'intentionalWalks': entry['stats'].get("intentionalWalks", 0),
                    'hits': entry['stats'].get("hits", 0),
                    'hitByPitch': entry['stats'].get("hitByPitch", 0),
                    'avg': entry['stats'].get("avg", 0),
                    'atBats': entry['stats'].get("atBats", 0),
                    'obp': entry['stats'].get("obp", 0),
                    'slg': entry['stats'].get("slg", 0),
                    'ops': entry['stats'].get("ops", 0),
                    'caughtStealing': entry['stats'].get("caughtStealing", 0),
                    'stolenBases': entry['stats'].get("stolenBases", 0),
                    'stolenBasePercentage': entry['stats'].get("stolenBasePercentage", 0),
                    'caughtStealingPercentage': entry['stats'].get("caughtStealingPercentage", 0),
                    'groundIntoDoublePlay': entry['stats'].get("groundIntoDoublePlay", 0),
                    'numberOfPitches': entry['stats'].get("numberOfPitches", 0),
                    'era': entry['stats'].get("era", 0),
                    'inningsPitched': entry['stats'].get("inningsPitched", 0),
                    'wins': entry['stats'].get("wins", 0),
                    'losses': entry['stats'].get("losses", 0),
                    'saves': entry['stats'].get("saves", 0),
                    'saveOpportunities': entry['stats'].get("saveOpportunities", 0),
                    'holds': entry['stats'].get("holds", 0),
                    'blownSaves': entry['stats'].get("blownSaves", 0),
                    'earnedRuns': entry['stats'].get("earnedRuns", 0),
                    'whip': entry['stats'].get("whip", 0),
                    'battersFaced': entry['stats'].get("battersFaced", 0),
                    'outs': entry['stats'].get("outs", 0),
                    'gamesPitched': entry['stats'].get("gamesPitched", 0),
                    'completeGames': entry['stats'].get("completeGames", 0),
                    'shutouts': entry['stats'].get("shutouts", 0),
                    'strikes': entry['stats'].get("strikes", 0),    
                    'strikePercentage': entry['stats'].get("strikePercentage", 0),
                    'hitBatsmen': entry['stats'].get("hitBatsmen", 0),
                    'balks': entry['stats'].get("balks", 0),
                    'wildPitches': entry['stats'].get("wildPitches", 0),
                    'pickoffs': entry['stats'].get("pickoffs", 0),
                    'totalBases': entry['stats'].get("totalBases", 0),
                    'groundOutsToAirouts': entry['stats'].get("groundOutsToAirouts", 0),
                    'winPercentage': entry['stats'].get("winPercentage", 0),
                    'pitchesPerInning': entry['stats'].get("pitchesPerInning", 0),
                    'gamesFinished': entry['stats'].get("gamesFinished", 0),
                    'strikeoutWalkRatio': entry['stats'].get("strikeoutWalkRatio", 0),
                    'strikeoutsPer9Inn': entry['stats'].get("strikeoutsPer9Inn", 0),
                    'walksPer9Inn': entry['stats'].get("walksPer9Inn", 0),
                    'hitsPer9Inn': entry['stats'].get("hitsPer9Inn", 0),
                    'runsScoredPer9': entry['stats'].get("runsScoredPer9", 0),
                    'homeRunsPer9': entry['stats'].get("homeRunsPer9", 0),
                    'inheritedRunners': entry['stats'].get("inheritedRunners", 0),
                    'inheritedRunnersScored': entry['stats'].get("inheritedRunnersScored", 0),
                    'catchersInterference': entry['stats'].get("catchersInterference", 0),
                    'sacBunts': entry['stats'].get("sacBunts", 0),
                    'sacFlies': entry['stats'].get("sacFlies", 0)
                })
        except Exception as e:
            logger.debug(f"Historical log error: player {player_id}: {e}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["id", "season"]).reset_index(drop=True)
    df.to_csv(f"./data/input/{today}_historical_pitching_data.csv", index=False)

    return df
