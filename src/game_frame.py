"""
Step 5: Game frame assembly.

Merges team_frame + pitcher_frame into a single training-ready dataframe.
Each row is one game with home team form, away team form, home SP features,
away SP features, and home-minus-away delta features.

Run from project root:
    python -m src.game_frame
"""
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import load_config, get_logger, DATA_DIR

logger = get_logger(__name__)

# Normalize team abbreviations so Statcast and Baseball-Reference agree.
# Keys = Statcast, values = Baseball-Reference.
STATCAST_TO_BBREF = {
    "TB": "TBR", "KC": "KCR", "SD": "SDP", "SF": "SFG", "WSH": "WSN",
    "CWS": "CHW", "AZ": "ARI", "ATH": "OAK",
}

# Columns from team_frame that become features (everything except identifiers/target)
TEAM_FEATURE_COLS = [
    "wins", "losses", "games_played", "win_pct", "days_since_previous_game",
    "win_streak", "loss_streak", "run_margin_last_game",
    "run_margin_season_avg", "runs_scored_season_avg", "runs_allowed_season_avg",
    "run_margin_last10_avg", "runs_scored_last10_avg", "runs_allowed_last10_avg", "wins_last10",
    "run_margin_last20_avg", "runs_scored_last20_avg", "runs_allowed_last20_avg", "wins_last20",
    "run_margin_last30_avg", "runs_scored_last30_avg", "runs_allowed_last30_avg", "wins_last30",
]

# Columns from pitcher_frame that become features
SP_FEATURE_COLS = [
    "sp_season_starts", "sp_days_rest", "sp_pitches_last_start", "sp_pitches_last5_total",
    "sp_season_era", "sp_season_whip", "sp_season_fip",
    "sp_season_k_pct", "sp_season_bb_pct", "sp_season_ip_per_start_avg",
    "sp_last3_era", "sp_last3_whip", "sp_last3_fip", "sp_last3_k_pct", "sp_last3_bb_pct", "sp_last3_ip_avg",
    "sp_last5_era", "sp_last5_whip", "sp_last5_fip", "sp_last5_k_pct", "sp_last5_bb_pct", "sp_last5_ip_avg",
]


def normalize_team_abbr(series: pd.Series) -> pd.Series:
    return series.replace(STATCAST_TO_BBREF)


def main():
    sport_cfg = load_config("sport")
    min_games = sport_cfg["sport"]["min_games_for_features"]

    logger.info("Loading source frames...")
    games = pd.read_csv(DATA_DIR / "raw" / "games_all.csv", parse_dates=["date"])
    team_frame = pd.read_csv(DATA_DIR / "features" / "team_frame.csv", parse_dates=["date"])
    pitcher_frame = pd.read_csv(DATA_DIR / "features" / "pitcher_frame.csv", parse_dates=["date"])
    logger.info(f"  {len(games):,} games | {len(team_frame):,} team-game rows | {len(pitcher_frame):,} pitcher starts")

    # Normalize Statcast team codes to Baseball-Reference codes
    pitcher_frame["team"] = normalize_team_abbr(pitcher_frame["team"])
    pitcher_frame["opponent"] = normalize_team_abbr(pitcher_frame["opponent"])

    # ----- Join home team features -----
    logger.info("Joining home team features...")
    home_tf = team_frame.rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in TEAM_FEATURE_COLS}}
    )[["date", "home_team"] + [f"home_{c}" for c in TEAM_FEATURE_COLS]]
    df = games.merge(home_tf, on=["date", "home_team"], how="left")

    # ----- Join away team features -----
    logger.info("Joining away team features...")
    away_tf = team_frame.rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in TEAM_FEATURE_COLS}}
    )[["date", "away_team"] + [f"away_{c}" for c in TEAM_FEATURE_COLS]]
    df = df.merge(away_tf, on=["date", "away_team"], how="left")

    # ----- Join starting pitchers + their features -----
    # Home SP: rows in pitcher_frame where is_home == 1 at that date/team
    logger.info("Joining home starting pitcher features...")
    home_sp = pitcher_frame[pitcher_frame["is_home"] == 1].copy()
    home_sp = home_sp.rename(
        columns={
            "team": "home_team",
            "pitcher_id": "home_sp_id",
            "pitcher_name": "home_sp_name",
            **{c: f"home_{c}" for c in SP_FEATURE_COLS},
        }
    )[["date", "home_team", "home_sp_id", "home_sp_name"] + [f"home_{c}" for c in SP_FEATURE_COLS]]
    df = df.merge(home_sp, on=["date", "home_team"], how="left")

    logger.info("Joining away starting pitcher features...")
    away_sp = pitcher_frame[pitcher_frame["is_home"] == 0].copy()
    away_sp = away_sp.rename(
        columns={
            "team": "away_team",
            "pitcher_id": "away_sp_id",
            "pitcher_name": "away_sp_name",
            **{c: f"away_{c}" for c in SP_FEATURE_COLS},
        }
    )[["date", "away_team", "away_sp_id", "away_sp_name"] + [f"away_{c}" for c in SP_FEATURE_COLS]]
    df = df.merge(away_sp, on=["date", "away_team"], how="left")

    logger.info(f"After joins: {len(df):,} rows")

    # ----- Drop games without identified starters -----
    before = len(df)
    df = df.dropna(subset=["home_sp_id", "away_sp_id"]).copy()
    logger.info(f"After dropping unidentified-starter games: {len(df):,} (-{before - len(df)})")

    # ----- Drop early-season games (rolling features too noisy) -----
    before = len(df)
    df = df[
        (df["home_games_played"] >= min_games) &
        (df["away_games_played"] >= min_games)
    ].copy()
    logger.info(f"After min_games={min_games} filter: {len(df):,} (-{before - len(df)})")

    # ----- Compute delta features (home - away) for every matched pair -----
    logger.info("Computing delta features (home - away)...")
    for c in TEAM_FEATURE_COLS + SP_FEATURE_COLS:
        hcol, acol = f"home_{c}", f"away_{c}"
        if hcol in df.columns and acol in df.columns:
            df[f"delta_{c}"] = df[hcol] - df[acol]

    # ----- Final tidy -----
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Final game frame: {len(df):,} rows, {len(df.columns)} columns")

    # Quick NaN report
    nan_counts = df.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if len(nan_counts) > 0:
        logger.info(f"Columns with NaN (top 10):")
        for col, n in nan_counts.head(10).items():
            logger.info(f"  {col}: {n} ({100*n/len(df):.1f}%)")

    out_path = DATA_DIR / "features" / "game_frame.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()