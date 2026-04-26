"""
Step 2: Data ingestion.

Pulls MLB game results + starting pitcher info for the configured seasons
using pybaseball, and saves raw CSVs to data/raw/.

Run from project root:
    python -m src.ingest
"""
from pathlib import Path
import pandas as pd
from pybaseball import schedule_and_record, cache

from src.utils import load_config, get_logger, DATA_DIR

# Cache pybaseball downloads so reruns are fast and friendly to servers
cache.enable()

logger = get_logger(__name__)

# MLB team abbreviations as used by Baseball-Reference (what pybaseball needs)
MLB_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]


def fetch_season_games(season: int) -> pd.DataFrame:
    """Pull every game for a season by iterating over each team's schedule.

    Baseball-Reference returns each game from both teams' perspectives, so
    we de-duplicate by keeping only rows where the team is the home team.
    """
    logger.info(f"Fetching {season} schedules for {len(MLB_TEAMS)} teams...")
    frames = []
    for i, team in enumerate(MLB_TEAMS, 1):
        try:
            df = schedule_and_record(season, team)
            df["team_for_row"] = team
            df["season"] = season
            frames.append(df)
            logger.info(f"  [{i}/{len(MLB_TEAMS)}] {team}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  {team} failed: {e}")
    season_df = pd.concat(frames, ignore_index=True)
    # De-duplicate: keep only rows where this team was home
    season_df = season_df[season_df["Home_Away"] == "Home"].copy()
    logger.info(f"Season {season}: {len(season_df)} games after dedup")
    return season_df


def normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and slim the raw schedule_and_record output into our schema."""
    out = pd.DataFrame({
        "season": df["season"],
        "date": pd.to_datetime(df["Date"].astype(str) + " " + df["season"].astype(str),
                                errors="coerce", format="%A, %b %d %Y"),
        "home_team": df["team_for_row"],
        "away_team": df["Opp"],
        "home_score": pd.to_numeric(df["R"], errors="coerce"),
        "away_score": pd.to_numeric(df["RA"], errors="coerce"),
    })
    # Drop games that haven't been played yet (scores missing)
    out = out.dropna(subset=["home_score", "away_score", "date"])
    out["home_team_won"] = (out["home_score"] > out["away_score"]).astype(int)
    return out.sort_values("date").reset_index(drop=True)


def main():
    sport_cfg = load_config("sport")
    seasons = sport_cfg["sport"]["seasons"]
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_games = []
    for season in seasons:
        raw = fetch_season_games(season)
        clean = normalize_games(raw)
        season_path = raw_dir / f"games_{season}.csv"
        clean.to_csv(season_path, index=False)
        logger.info(f"Saved {len(clean)} games to {season_path}")
        all_games.append(clean)

    combined = pd.concat(all_games, ignore_index=True)
    combined_path = raw_dir / "games_all.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(f"Saved combined {len(combined)} games to {combined_path}")


if __name__ == "__main__":
    main()