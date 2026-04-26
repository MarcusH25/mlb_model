"""
Step 4a: Statcast ingestion.

Pulls pitch-level Statcast data for each configured season, identifies the
starting pitchers per game, and aggregates per-start outcome stats (IP, K, BB,
HBP, H, HR, fly balls, runs allowed, derived FIP/WHIP/ERA).

Raw pitch data is saved as parquet (compact). The aggregated pitcher_starts
table is what Step 4b consumes.

Run from project root:
    python -m src.ingest_statcast
"""
from pathlib import Path
import pandas as pd
import numpy as np
from pybaseball import statcast, cache

from src.utils import load_config, get_logger, DATA_DIR

cache.enable()
logger = get_logger(__name__)

# --- Event classification for aggregating pitch-level data to per-start stats ---
OUT_EVENTS = {
    'strikeout', 'strikeout_double_play', 'field_out', 'force_out',
    'grounded_into_double_play', 'double_play', 'triple_play',
    'sac_fly', 'sac_fly_double_play', 'sac_bunt', 'sac_bunt_double_play',
    'fielders_choice_out', 'other_out',
}
K_EVENTS = {'strikeout', 'strikeout_double_play'}
BB_EVENTS = {'walk'}
HBP_EVENTS = {'hit_by_pitch'}
HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}
HR_EVENTS = {'home_run'}
DOUBLE_PLAY_EVENTS = {
    'strikeout_double_play', 'grounded_into_double_play', 'double_play',
    'sac_fly_double_play', 'sac_bunt_double_play',
}
TRIPLE_PLAY_EVENTS = {'triple_play'}

FIP_CONSTANT = 3.10  # approximates league-avg FIP adjustment


def download_season_statcast(season: int, raw_dir: Path) -> Path:
    """Download a full season of Statcast pitch data, save as parquet."""
    out_path = raw_dir / f"statcast_{season}.parquet"
    if out_path.exists():
        logger.info(f"  {out_path.name} already exists, skipping download")
        return out_path

    start = f"{season}-03-15"
    end = f"{season}-11-01"
    logger.info(f"  Downloading Statcast {start} to {end}...")
    df = statcast(start_dt=start, end_dt=end)
    logger.info(f"  Downloaded {len(df)} pitches, saving to {out_path.name}")
    df.to_parquet(out_path, index=False)
    return out_path


def identify_starting_pitchers(pitches: pd.DataFrame) -> pd.DataFrame:
    """For each game, find the first pitcher each team used (the starter).

    Top of inning -> away batting, home pitching.
    Bottom of inning -> home batting, away pitching.
    """
    first_pitches = (
        pitches
        .sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        .groupby(['game_pk', 'inning_topbot'], as_index=False)
        .first()
    )

    rows = []
    for _, r in first_pitches.iterrows():
        if r['inning_topbot'] == 'Top':
            team, opp, is_home = r['home_team'], r['away_team'], 1
        else:
            team, opp, is_home = r['away_team'], r['home_team'], 0
        rows.append({
            'game_pk': r['game_pk'],
            'date': r['game_date'],
            'pitcher_id': r['pitcher'],
            'pitcher_name': r['player_name'],
            'team': team,
            'opponent': opp,
            'is_home': is_home,
        })
    return pd.DataFrame(rows)


def aggregate_pitcher_starts(pitches: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level data to per-start outcome stats for each starter."""
    # Keep only pitches thrown by identified starters
    sp_pitches = pitches.merge(
        starters[['game_pk', 'pitcher_id']].rename(columns={'pitcher_id': 'pitcher'}),
        on=['game_pk', 'pitcher'],
        how='inner',
    )

    rows = []
    for (game_pk, pitcher_id), grp in sp_pitches.groupby(['game_pk', 'pitcher']):
        events = grp['events'].dropna()

        # Outs (most events = 1 out; double play = 2; triple play = 3)
        outs = (events.isin(OUT_EVENTS)).sum()
        outs += (events.isin(DOUBLE_PLAY_EVENTS)).sum()  # +1 extra
        outs += 2 * (events.isin(TRIPLE_PLAY_EVENTS)).sum()  # +2 extra

        k = (events.isin(K_EVENTS)).sum()
        bb = (events.isin(BB_EVENTS)).sum()
        hbp = (events.isin(HBP_EVENTS)).sum()
        hits = (events.isin(HIT_EVENTS)).sum()
        hr = (events.isin(HR_EVENTS)).sum()
        bf = grp['at_bat_number'].nunique()
        pitches_thrown = len(grp)

        # Fly balls: launch angle > 25° on contact events
        contact = grp[grp['events'].notna()]
        fly_balls = ((contact['launch_angle'] > 25)).sum()

        # Runs allowed while starter was in (proxy for ER — slight overestimate)
        ab_last = grp.drop_duplicates('at_bat_number', keep='last')
        if 'post_bat_score' in grp.columns and 'bat_score' in grp.columns:
            runs_allowed = int(
                (ab_last['post_bat_score'] - ab_last['bat_score']).clip(lower=0).sum()
            )
        else:
            runs_allowed = 0

        rows.append({
            'game_pk': game_pk,
            'pitcher_id': pitcher_id,
            'outs_recorded': int(outs),
            'ip': outs / 3.0,
            'batters_faced': bf,
            'pitches_thrown': pitches_thrown,
            'strikeouts': int(k),
            'walks': int(bb),
            'hbp': int(hbp),
            'hits': int(hits),
            'home_runs': int(hr),
            'fly_balls': int(fly_balls),
            'runs_allowed': runs_allowed,
        })

    agg = pd.DataFrame(rows)
    out = starters.merge(agg, on=['game_pk', 'pitcher_id'], how='inner')

    # Derived rate / context stats
    out['k_pct'] = np.where(out['batters_faced'] > 0, out['strikeouts'] / out['batters_faced'], np.nan)
    out['bb_pct'] = np.where(out['batters_faced'] > 0, out['walks'] / out['batters_faced'], np.nan)
    out['fip'] = np.where(
        out['ip'] > 0,
        ((13 * out['home_runs'] + 3 * (out['walks'] + out['hbp']) - 2 * out['strikeouts']) / out['ip']) + FIP_CONSTANT,
        np.nan,
    )
    out['whip'] = np.where(out['ip'] > 0, (out['walks'] + out['hits']) / out['ip'], np.nan)
    out['era'] = np.where(out['ip'] > 0, 9 * out['runs_allowed'] / out['ip'], np.nan)

    out['date'] = pd.to_datetime(out['date'])
    out['season'] = out['date'].dt.year
    return out


def main():
    sport_cfg = load_config("sport")
    seasons = sport_cfg["sport"]["seasons"]
    raw_dir = DATA_DIR / "raw"
    processed_dir = DATA_DIR / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_starts = []
    for season in seasons:
        logger.info(f"=== Season {season} ===")
        parquet_path = download_season_statcast(season, raw_dir)

        logger.info(f"  Loading {parquet_path.name}...")
        pitches = pd.read_parquet(parquet_path)
        logger.info(f"  Loaded {len(pitches):,} pitches")

        # Filter to regular season
        pitches['game_date'] = pd.to_datetime(pitches['game_date'])
        pitches = pitches[
            (pitches['game_date'] >= f"{season}-03-28") &
            (pitches['game_date'] <= f"{season}-10-01")
        ]
        logger.info(f"  After regular-season filter: {len(pitches):,} pitches")

        logger.info("  Identifying starting pitchers...")
        starters = identify_starting_pitchers(pitches)
        logger.info(f"  Found {len(starters):,} starter-game records")

        logger.info("  Aggregating pitcher-start stats...")
        starts = aggregate_pitcher_starts(pitches, starters)
        logger.info(f"  Built {len(starts):,} pitcher starts for {season}")
        all_starts.append(starts)

        del pitches  # free memory before next season

    combined = (
        pd.concat(all_starts, ignore_index=True)
        .sort_values('date')
        .reset_index(drop=True)
    )
    out_path = processed_dir / "pitcher_starts.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"Saved {len(combined):,} pitcher starts to {out_path}")


if __name__ == "__main__":
    main()