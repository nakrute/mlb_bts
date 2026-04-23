"""
Create sample training data for testing the pipeline.
This generates training data with lagged features as expected by the model.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import DATA_DIR, ROLLING_WINDOWS

def create_sample_training_data():
    """Generate realistic sample training data with lagged features."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    all_rows = []
    np.random.seed(42)
    
    for player_id in range(100, 150):  # 50 sample players
        baseline_avg = np.random.uniform(0.250, 0.310)
        
        season_start = datetime(2023, 3, 28)
        player_games = []
        
        # First, generate all non-zero AB games
        for game_num in range(200):
            game_date = season_start + timedelta(days=game_num)
            ab = np.random.choice([3, 4, 5], p=[0.25, 0.60, 0.15])
            hits = np.random.binomial(ab, baseline_avg)
            got_hit = 1 if hits >= 1 else 0
            
            player_games.append({
                "date": game_date,
                "ab": ab,
                "hits": hits,
                "so": np.random.binomial(ab, 0.20),
                "bb": np.random.binomial(ab, 0.08),
                "got_hit": got_hit,
            })
        
        # Now build lagged rows starting from game 20 (need 20 prior games)
        for i in range(20, len(player_games)):
            prior_games = player_games[:i]
            current_game = player_games[i]
            
            row = {
                "player_id": player_id,
                "player_name": f"Player_{player_id}",
                "season": 2023,
                "date": current_game["date"],
                "got_hit": current_game["got_hit"],
            }
            
            # Add lagged features for each rolling window
            for window in ROLLING_WINDOWS:
                recent = prior_games[-window:] if len(prior_games) >= window else prior_games
                if not recent:
                    continue
                    
                total_ab = sum(g["ab"] for g in recent)
                total_hits = sum(g["hits"] for g in recent)
                games_hit = sum(g["got_hit"] for g in recent)
                total_games = len(recent)
                
                if total_ab > 0:
                    row[f"roll{window}_avg"] = total_hits / total_ab
                    row[f"roll{window}_k_pct"] = sum(g["so"] for g in recent) / total_ab
                    row[f"roll{window}_bb_pct"] = sum(g["bb"] for g in recent) / total_ab
                else:
                    row[f"roll{window}_avg"] = 0.250
                    row[f"roll{window}_k_pct"] = 0.20
                    row[f"roll{window}_bb_pct"] = 0.08
                
                row[f"roll{window}_hit_game_rate"] = games_hit / total_games if total_games > 0 else 0
                row[f"roll{window}_ab_per_game"] = total_ab / total_games if total_games > 0 else 0
            
            # Add streak and recent hit games
            streak = 0
            for game in reversed(prior_games):
                if game["got_hit"] == 1:
                    streak += 1
                else:
                    break
            row["current_streak"] = streak
            row["last5_hit_games"] = sum(g["got_hit"] for g in prior_games[-5:])
            row["last10_hit_games"] = sum(g["got_hit"] for g in prior_games[-10:])
            
            # Season cumulative
            row["season_pa"] = sum(g["ab"] for g in prior_games)
            row["season_hits"] = sum(g["hits"] for g in prior_games)
            season_ab = sum(g["ab"] for g in prior_games)
            row["season_avg"] = sum(g["hits"] for g in prior_games) / season_ab if season_ab > 0 else 0.250
            row["season_games"] = len(prior_games)
            
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    
    cache_path = os.path.join(DATA_DIR, "training_data.parquet")
    df.to_parquet(cache_path, index=False)
    
    print(f"Created {cache_path}")
    print(f"Total rows: {len(df)}")
    print(f"Unique players: {df['player_id'].nunique()}")
    print(f"Hit rate: {df['got_hit'].mean():.3f}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    create_sample_training_data()
