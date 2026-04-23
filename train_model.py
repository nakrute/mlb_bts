import logging
import warnings
import pandas as pd

import data_fetcher as df_get

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
today = pd.to_datetime("today").normalize().strftime("%Y-%m-%d")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("BTS MODEL — TRAINING PIPELINE")
    logger.info("=" * 60)
    df = df_get.historical_data_by_today_players(today)
    pitchers = df_get.get_historical_pitching_logs(df)
    print(df)
    print(pitchers)
    logger.info("Training complete. You can now run: python daily_picks.py")