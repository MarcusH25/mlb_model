"""
Step 10-1: Park factor features.

Adds per-game park run factors based on published multi-year averages.
Values are stable across seasons and represent pre-game known information,
so no leakage risk.

Source: FanGraphs 3-year rolling park factors (100 = neutral, >100 means
more runs score there, <100 means fewer). We use 2021-2023 averages
applied to all seasons — for research-grade analysis, using stable
historical averages rather than year-specific factors prevents leakage.

Run from project root:
    python -m src.park_factors
"""
from pathlib import Path
import pandas as pd

from src.utils import get_logger, DATA_DIR

logger = get_logger(__name__)

# Park run factors (100 = neutral).
# Sourced from FanGraphs 3-year rolling averages 2021-2023.
# Keyed by BBRef home team abbreviation.
PARK_FACTORS = {
    "COL": 112,  # Coors Field — highest run environment
    "CIN": 106,  # Great American
    "BOS": 105,  # Fenway
    "KCR": 104,  # Kauffman
    "TEX": 103,  # Globe Life
    "PHI": 102,  # Citizens Bank
    "TOR": 102,  # Rogers Centre
    "ATL": 101,  # Truist
    "BAL": 101,  # Camden Yards
    "WSN": 101,  # Nationals Park
    "MIN": 101,  # Target Field
    "ARI": 100,  # Chase Field
    "HOU": 100,  # Minute Maid
    "LAA":  99,  # Angel Stadium
    "MIL":  99,  # American Family Field
    "CHC":  99,  # Wrigley
    "TBR":  99,  # Tropicana
    "NYM":  99,  # Citi Field
    "NYY":  98,  # Yankee Stadium
    "CHW":  98,  # Guaranteed Rate
    "CLE":  98,  # Progressive
    "DET":  97,  # Comerica
    "STL":  97,  # Busch
    "SEA":  95,  # T-Mobile
    "PIT":  95,  # PNC
    "LAD":  94,  # Dodger Stadium
    "OAK":  94,  # Coliseum
    "MIA":  93,  # LoanDepot
    "SFG":  92,  # Oracle
    "SDP":  92,  # Petco — lowest run environment
}


def add_park_factors(gf: pd.DataFrame) -> pd.DataFrame:
    """Adds home_park_factor as a feature. Normalizes 100 -> 1.00."""
    gf = gf.copy()
    # Normalize so 1.00 = neutral park (easier for model to learn)
    gf["home_park_factor"] = gf["home_team"].map(PARK_FACTORS) / 100.0

    missing = gf[gf["home_park_factor"].isna()]["home_team"].unique().tolist()
    if len(missing) > 0:
        logger.warning(f"No park factor for teams: {missing} (filling with 1.00)")
        gf["home_park_factor"] = gf["home_park_factor"].fillna(1.00)

    return gf


def main():
    in_path = DATA_DIR / "features" / "game_frame.csv"
    gf = pd.read_csv(in_path, parse_dates=["date"])
    logger.info(f"Loaded game frame: {len(gf):,} rows, {len(gf.columns)} cols")

    gf = add_park_factors(gf)
    logger.info(f"Added park factor. Distribution:")
    logger.info(f"  min={gf['home_park_factor'].min():.2f}")
    logger.info(f"  median={gf['home_park_factor'].median():.2f}")
    logger.info(f"  max={gf['home_park_factor'].max():.2f}")

    # Save — overwrites the existing game_frame since park factor becomes baseline
    gf.to_csv(in_path, index=False)
    logger.info(f"Saved back to {in_path} with {len(gf.columns)} columns")


if __name__ == "__main__":
    main()