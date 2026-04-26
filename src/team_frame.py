"""
Step 3: Team frame builder.

For each team in each season, build a time-ordered dataframe of the team's
rolling form features. Each row represents one game, and all features are
computed using ONLY games that occurred before that game (no leakage).

This mirrors AlphaPy's generate_team_frame() but adapted for MLB:
  - 'runs' instead of 'points'
  - No spread/over-under features (moneyline only)
  - Multiple rolling windows instead of one
  - Home/away separation when computing the per-team timeline

Run from project root:
    python -m src.team_frame
"""
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import load_config, get_logger, DATA_DIR

logger = get_logger(__name__)


def build_team_timeline(games: pd.DataFrame, team: str) -> pd.DataFrame:
    """Extract a team's full season of games in chronological order.

    For each game the team played, record whether they were home/away, their
    runs scored, runs allowed, and the resulting win/loss.
    """
    home_games = games[games["home_team"] == team].copy()
    home_games["is_home"] = 1
    home_games["team"] = team
    home_games["opponent"] = home_games["away_team"]
    home_games["runs_scored"] = home_games["home_score"]
    home_games["runs_allowed"] = home_games["away_score"]

    away_games = games[games["away_team"] == team].copy()
    away_games["is_home"] = 0
    away_games["team"] = team
    away_games["opponent"] = away_games["home_team"]
    away_games["runs_scored"] = away_games["away_score"]
    away_games["runs_allowed"] = away_games["home_score"]

    tl = pd.concat([home_games, away_games], ignore_index=True)
    tl = tl.sort_values("date").reset_index(drop=True)

    tl["run_margin"] = tl["runs_scored"] - tl["runs_allowed"]
    tl["won"] = (tl["run_margin"] > 0).astype(int)
    tl["lost"] = (tl["run_margin"] < 0).astype(int)

    return tl[[
        "season", "date", "team", "opponent", "is_home",
        "runs_scored", "runs_allowed", "run_margin", "won", "lost",
    ]]


def add_rolling_features(tl: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add pre-game rolling form features to a team's timeline.

    Critical: every feature is lagged by one game via .shift(1) so we use
    only information available BEFORE the current game. This prevents leakage.
    """
    tl = tl.copy()

    # Season-to-date (expanding) — wins/losses BEFORE this game
    tl["wins"] = tl["won"].shift(1).expanding().sum()
    tl["losses"] = tl["lost"].shift(1).expanding().sum()
    tl["games_played"] = tl["wins"] + tl["losses"]
    tl["win_pct"] = np.where(
        tl["games_played"] > 0, tl["wins"] / tl["games_played"], np.nan
    )

    # Rest: days since previous game
    tl["days_since_previous_game"] = (
        tl["date"] - tl["date"].shift(1)
    ).dt.days

    # Streaks — current run of W or L going into this game
    won_shift = tl["won"].shift(1)
    # group consecutive identical values, then count within each group
    streak_group = (won_shift != won_shift.shift()).cumsum()
    streak_len = won_shift.groupby(streak_group).cumcount() + 1
    tl["win_streak"] = np.where(won_shift == 1, streak_len, 0)
    tl["loss_streak"] = np.where(won_shift == 0, streak_len, 0)

    # Last-game features
    tl["run_margin_last_game"] = tl["run_margin"].shift(1)

    # Season-to-date averages (expanding)
    tl["run_margin_season_avg"] = tl["run_margin"].shift(1).expanding().mean()
    tl["runs_scored_season_avg"] = tl["runs_scored"].shift(1).expanding().mean()
    tl["runs_allowed_season_avg"] = tl["runs_allowed"].shift(1).expanding().mean()

    # Rolling windows (10, 20, 30 games)
    for w in windows:
        tl[f"run_margin_last{w}_avg"] = (
            tl["run_margin"].shift(1).rolling(window=w, min_periods=1).mean()
        )
        tl[f"runs_scored_last{w}_avg"] = (
            tl["runs_scored"].shift(1).rolling(window=w, min_periods=1).mean()
        )
        tl[f"runs_allowed_last{w}_avg"] = (
            tl["runs_allowed"].shift(1).rolling(window=w, min_periods=1).mean()
        )
        tl[f"wins_last{w}"] = (
            tl["won"].shift(1).rolling(window=w, min_periods=1).sum()
        )

    # Clean streak NaN on row 0
    tl["win_streak"] = tl["win_streak"].fillna(0).astype(int)
    tl["loss_streak"] = tl["loss_streak"].fillna(0).astype(int)

    return tl


def build_all_team_frames(games: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """For each (team, season) combination, build a full feature timeline.

    Rebuilding per-season ensures rolling stats don't leak across seasons.
    """
    all_frames = []
    seasons = sorted(games["season"].unique())
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))

    for season in seasons:
        season_games = games[games["season"] == season]
        for team in teams:
            tl = build_team_timeline(season_games, team)
            if len(tl) == 0:
                continue
            tl = add_rolling_features(tl, windows)
            all_frames.append(tl)

    out = pd.concat(all_frames, ignore_index=True)
    return out


def main():
    sport_cfg = load_config("sport")
    windows = sport_cfg["sport"]["rolling_windows"]

    logger.info(f"Loading games from data/raw/games_all.csv")
    games = pd.read_csv(DATA_DIR / "raw" / "games_all.csv", parse_dates=["date"])
    logger.info(f"Loaded {len(games)} games")

    logger.info(f"Building team frames with rolling windows: {windows}")
    team_frame = build_all_team_frames(games, windows)
    logger.info(f"Built team frame: {len(team_frame)} rows, {len(team_frame.columns)} columns")

    out_path = DATA_DIR / "features" / "team_frame.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    team_frame.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()