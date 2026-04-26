"""
Step 4b: Pitcher frame builder.

For each starting pitcher, build a time-ordered dataframe of their
pre-start rolling form features. Mirrors the team_frame pattern: shift
by one start so features reflect only prior starts (no leakage).

Run from project root:
    python -m src.pitcher_frame
"""
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import get_logger, DATA_DIR

logger = get_logger(__name__)

ROLLING_STARTS = [3, 5]


def add_rolling_features(pf: pd.DataFrame) -> pd.DataFrame:
    """Add pre-start rolling features for a single pitcher's timeline.

    pf is already sorted by date and belongs to one pitcher in one season.
    """
    pf = pf.copy()

    # Counters / rest
    pf["sp_season_starts"] = np.arange(len(pf))  # 0-indexed: starts BEFORE this one
    pf["sp_days_rest"] = (pf["date"] - pf["date"].shift(1)).dt.days
    pf["sp_pitches_last_start"] = pf["pitches_thrown"].shift(1)
    pf["sp_pitches_last5_total"] = (
        pf["pitches_thrown"].shift(1).rolling(window=5, min_periods=1).sum()
    )

    # Season-to-date (expanding) — weighted correctly by IP
    # Use cumulative sums then divide (so ERA = 9 * cum_runs / cum_ip, not naive avg of per-start ERAs)
    cum_ip = pf["ip"].shift(1).cumsum()
    cum_er = pf["runs_allowed"].shift(1).cumsum()
    cum_h = pf["hits"].shift(1).cumsum()
    cum_bb = pf["walks"].shift(1).cumsum()
    cum_hbp = pf["hbp"].shift(1).cumsum()
    cum_hr = pf["home_runs"].shift(1).cumsum()
    cum_k = pf["strikeouts"].shift(1).cumsum()
    cum_bf = pf["batters_faced"].shift(1).cumsum()

    pf["sp_season_era"] = np.where(cum_ip > 0, 9.0 * cum_er / cum_ip, np.nan)
    pf["sp_season_whip"] = np.where(cum_ip > 0, (cum_bb + cum_h) / cum_ip, np.nan)
    pf["sp_season_fip"] = np.where(
        cum_ip > 0,
        ((13 * cum_hr + 3 * (cum_bb + cum_hbp) - 2 * cum_k) / cum_ip) + 3.10,
        np.nan,
    )
    pf["sp_season_k_pct"] = np.where(cum_bf > 0, cum_k / cum_bf, np.nan)
    pf["sp_season_bb_pct"] = np.where(cum_bf > 0, cum_bb / cum_bf, np.nan)
    pf["sp_season_ip_per_start_avg"] = pf["ip"].shift(1).expanding().mean()

    # Rolling windows over last N starts — same IP-weighted approach
    for w in ROLLING_STARTS:
        roll_ip = pf["ip"].shift(1).rolling(window=w, min_periods=1).sum()
        roll_er = pf["runs_allowed"].shift(1).rolling(window=w, min_periods=1).sum()
        roll_h = pf["hits"].shift(1).rolling(window=w, min_periods=1).sum()
        roll_bb = pf["walks"].shift(1).rolling(window=w, min_periods=1).sum()
        roll_hbp = pf["hbp"].shift(1).rolling(window=w, min_periods=1).sum()
        roll_hr = pf["home_runs"].shift(1).rolling(window=w, min_periods=1).sum()
        roll_k = pf["strikeouts"].shift(1).rolling(window=w, min_periods=1).sum()
        roll_bf = pf["batters_faced"].shift(1).rolling(window=w, min_periods=1).sum()

        pf[f"sp_last{w}_era"] = np.where(roll_ip > 0, 9.0 * roll_er / roll_ip, np.nan)
        pf[f"sp_last{w}_whip"] = np.where(roll_ip > 0, (roll_bb + roll_h) / roll_ip, np.nan)
        pf[f"sp_last{w}_fip"] = np.where(
            roll_ip > 0,
            ((13 * roll_hr + 3 * (roll_bb + roll_hbp) - 2 * roll_k) / roll_ip) + 3.10,
            np.nan,
        )
        pf[f"sp_last{w}_k_pct"] = np.where(roll_bf > 0, roll_k / roll_bf, np.nan)
        pf[f"sp_last{w}_bb_pct"] = np.where(roll_bf > 0, roll_bb / roll_bf, np.nan)
        pf[f"sp_last{w}_ip_avg"] = pf["ip"].shift(1).rolling(window=w, min_periods=1).mean()

    return pf


def build_pitcher_frame(starts: pd.DataFrame) -> pd.DataFrame:
    """Build rolling features for all (pitcher, season) timelines."""
    starts = starts.sort_values(["pitcher_id", "season", "date"]).reset_index(drop=True)
    frames = []
    for (pitcher_id, season), grp in starts.groupby(["pitcher_id", "season"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        frames.append(add_rolling_features(grp))
    return pd.concat(frames, ignore_index=True)


def main():
    logger.info("Loading pitcher_starts.csv")
    starts = pd.read_csv(
        DATA_DIR / "processed" / "pitcher_starts.csv", parse_dates=["date"]
    )
    logger.info(f"Loaded {len(starts):,} starts, {starts['pitcher_id'].nunique()} pitchers")

    logger.info("Building pitcher frame (rolling features per pitcher-season)...")
    pf = build_pitcher_frame(starts)
    logger.info(f"Built pitcher frame: {len(pf):,} rows, {len(pf.columns)} columns")

    out_path = DATA_DIR / "features" / "pitcher_frame.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pf.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()