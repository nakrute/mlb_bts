# config.py — Central configuration for BTS Model
import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# API Keys (set in .env file)
# ──────────────────────────────────────────────
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")  # Optional: https://openweathermap.org/api

# ──────────────────────────────────────────────
# Model Parameters
# ──────────────────────────────────────────────
MODEL_TYPE = "xgboost"          # "xgboost" or "lightgbm"
MIN_HIT_PROBABILITY = 0.68      # Minimum P(hit) to consider a pick
TOP_N_PICKS = 2                 # BTS allows 2 picks per day
CONFIDENCE_THRESHOLD = 0.72     # Flag picks above this as "high confidence"

# ──────────────────────────────────────────────
# Feature Windows
# ──────────────────────────────────────────────
ROLLING_WINDOWS = [7, 14, 30]   # Days for rolling batting stats
MIN_AB_FOR_PLATOON = 20         # Min AB to use platoon splits
MIN_AB_H2H = 10                 # Min AB vs pitcher for head-to-head stats

# ──────────────────────────────────────────────
# Training Data
# ──────────────────────────────────────────────
TRAINING_START_YEAR = 2018
TRAINING_END_YEAR = 2023
DATA_DIR = "data"
MODEL_DIR = "models"
PICKS_DIR = "picks"
LOGS_DIR = "logs"

# ──────────────────────────────────────────────
# Ballpark Run Factors (2023, 100 = neutral)
# Higher = more offense = better for hitters
# ──────────────────────────────────────────────
PARK_FACTORS = {
    "COL": 115,  # Coors Field — massive hitter park
    "CIN": 107,  # Great American Ball Park
    "BOS": 106,  # Fenway Park
    "TEX": 105,  # Globe Life Field
    "PHI": 104,  # Citizens Bank Park
    "BAL": 103,  # Camden Yards
    "MIL": 102,  # American Family Field
    "NYY": 101,  # Yankee Stadium
    "CHC": 101,  # Wrigley Field
    "ATL": 100,
    "HOU": 100,
    "LAD": 99,
    "NYM": 99,
    "MIN": 99,
    "CLE": 98,
    "DET": 98,
    "TOR": 98,
    "SEA": 97,   # T-Mobile Park — pitcher friendly
    "SF":  97,   # Oracle Park
    "OAK": 97,
    "MIA": 96,
    "TB":  96,   # Tropicana Field
    "LAA": 100,
    "WSH": 100,
    "STL": 100,
    "KC":  100,
    "PIT": 100,
    "ARI": 100,
    "SD":  97,   # Petco Park — pitcher friendly
    "CWS": 100,
}

TEAM_NAMES = [
    "Arizona Diamondbacks",
    "Oakland Athletics",
    "Atlanta Braves",
    "Baltimore Orioles",
    "Boston Red Sox",
    "Chicago Cubs",
    "Chicago White Sox",
    "Cincinnati Reds",
    "Cleveland Guardians",
    "Colorado Rockies",
    "Detroit Tigers",
    "Houston Astros",
    "Kansas City Royals",
    "Los Angeles Angels",
    "Los Angeles Dodgers",
    "Miami Marlins",
    "Milwaukee Brewers",
    "Minnesota Twins",
    "New York Mets",
    "New York Yankees",
    "Philadelphia Phillies",
    "Pittsburgh Pirates",
    "San Diego Padres",
    "San Francisco Giants",
    "Seattle Mariners",
    "St. Louis Cardinals",
    "Tampa Bay Rays",
    "Texas Rangers",
    "Toronto Blue Jays",
    "Washington Nationals",
]


TRAIN_FEATURES = [
    "roll7_avg", "roll7_hit_game_rate", "roll7_ab_per_game", "roll7_k_pct", "roll7_bb_pct",
    "roll14_avg", "roll14_hit_game_rate", "roll14_ab_per_game",
    "roll30_avg", "roll30_hit_game_rate", "roll30_ab_per_game",
    "current_streak", "last5_hit_games", "last10_hit_games",
    "season_avg", "season_pa", "season_games",
]

# ──────────────────────────────────────────────
# Filters — avoid these situations
# ──────────────────────────────────────────────
MIN_LINEUP_POSITION = 1         # Ignore players batting lower than this (1=leadoff ... 9=ninth)
MAX_LINEUP_POSITION = 7         # Don't pick 8/9 hitters (fewer PA)
MIN_SEASON_PA = 50              # Ignore players with very few PA (sample size)
MIN_CAREER_GAMES = 100          # Avoid rookies with no track record

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = ""  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

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

DOME_STADIUMS = {
    "Tropicana Field", "loanDepot park", "Rogers Centre",
    "Globe Life Field", "Minute Maid Park", "T-Mobile Park",  # retractable
    "American Family Field", "Chase Field",
}


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