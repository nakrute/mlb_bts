import logging
import time

import pandas as pd
import statsapi

logger = logging.getLogger(__name__)
today = pd.to_datetime("today").normalize().strftime("%Y-%m-%d")


def get_todays_games(today: str=today) -> pd.DataFrame:
    games = statsapi.schedule()
    games_df = pd.DataFrame(games) 
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
    pd.DataFrame(player_id_set, columns=["player_id", "game_id"]).to_csv(f"./data/input/{today}_players_games.csv", index=False)
    player_ids = [x[0] for x in player_id_set]
    logger.info(f"Found {len(player_ids)} unique players in today's games")
    
    all_df = pd.DataFrame()
    for count, pid in enumerate(player_ids, start=1):
        try:
            logging.info(f"Processing player {pid} ({count}/{len(player_ids)})")
            rows = get_historical_batting_logs(pid)
            if not rows.empty:
                all_df = pd.concat([all_df, rows], ignore_index=True)
        except Exception as e:
            logger.debug(f"Error fetching historical data for player {pid}: {e}")
            pass
    
    all_df.to_csv(f"./data/input/{today}_historical_data.csv", index=False)
    logger.info(f"Built historical dataset with {len(all_df)} rows for today's players")
    return all_df


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
                "name": f"{log.get('first_name', '')} {log.get('last_name', '')}",
                "position": log.get("position", ""),
                "season": entry.get("season", ""),
                "avg": entry['stats'].get("avg", 0),
                "hits": entry['stats'].get("hits", 0),
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


def get_historical_pitching_logs(player_df: pd.DataFrame, 
                                 today: str=today) -> pd.DataFrame:
    pitcher_df = player_df[player_df['position'] == "P"]
    pitcher_ids = pitcher_df['id'].unique()
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
                    "gamesPlayed": entry['stats'].get("gamesPlayed", 0),
                    "gamesStarted": entry['stats'].get("gamesStarted", 0),
                    "hits": entry['stats'].get("hits", 0),
                    "era": entry['stats'].get("era", 0),
                    "whip": entry['stats'].get("whip", 0),
                    "shutouts": entry['stats'].get("shutouts", 0),
                    "walksPer9Inn": entry['stats'].get("walksPer9Inn", 0),
                    "homeRunsPer9": entry['stats'].get("homeRunsPer9", 0),
                })
        except Exception as e:
            logger.debug(f"Historical log error: player {player_id}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(f"./data/input/{today}_historical_pitching_data.csv", index=False)

    return df
