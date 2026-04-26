"""
Step 8b-1: Ingest historical MLB moneyline odds.

Reads the mlb_odds_dataset.json from ArnavSaraogi and produces a clean CSV
with per-game closing moneylines and no-vig market probabilities.

Cleaning strategy (validated against 10,988 games):
  - Per-book trash filter: reject any book where |ML| > 800 OR |ML| < 100
    (catches post-game "settled" garbage AND nonsensical sub-100 values
    that break the profit calculation)
  - Per-book probability conversion FIRST, then aggregate median
    (avoids math discontinuity when American odds approach 0)
  - Derive clean consensus American odds FROM the no-vig probabilities
    plus a realistic 4.5% vig, so betting math always works

Run from project root:
    python -m src.ingest_odds
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils import get_logger, DATA_DIR

logger = get_logger(__name__)

MAX_ABS_ML = 800     # bigger than this = scraping trash
MIN_BOOKS = 2        # require at least 2 clean books
MARKET_VIG = 0.045   # typical US sportsbook: ~4.5% vig on MLB moneylines


def american_to_prob(odds: float) -> float | None:
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def _prob_to_american(p: float) -> int:
    if p >= 0.5:
        return int(round(-100.0 * p / (1 - p)))
    return int(round(100.0 * (1 - p) / p))


def parse_game(game: dict) -> dict | None:
    """Extract clean no-vig closing probabilities + realistic consensus odds."""
    mls = game.get("odds", {}).get("moneyline", [])
    if not mls:
        return None

    h_probs, a_probs = [], []
    for ml in mls:
        cl = ml.get("currentLine")
        if not cl:
            continue
        h, a = cl.get("homeOdds"), cl.get("awayOdds")
        if h is None or a is None or h == 0 or a == 0:
            continue
        if abs(h) > MAX_ABS_ML or abs(a) > MAX_ABS_ML:
            continue
        # REJECT sub-100 "odds" — mathematically nonsense in American format
        if abs(h) < 100 or abs(a) < 100:
            continue
        ph, pa = american_to_prob(h), american_to_prob(a)
        if ph is None or pa is None:
            continue
        total = ph + pa
        if total <= 0:
            continue
        h_probs.append(ph / total)
        a_probs.append(pa / total)

    if len(h_probs) < MIN_BOOKS:
        return None

    p_home_no_vig = float(np.median(h_probs))
    p_away_no_vig = float(np.median(a_probs))

    # Re-derive realistic American odds from the no-vig probability + standard vig
    half_vig = MARKET_VIG / 2.0
    p_home_vigged = min(max(p_home_no_vig * (1 + half_vig), 0.01), 0.99)
    p_away_vigged = min(max(p_away_no_vig * (1 + half_vig), 0.01), 0.99)

    return {
        "p_home_market": p_home_no_vig,
        "p_away_market": p_away_no_vig,
        "home_ml_consensus": _prob_to_american(p_home_vigged),
        "away_ml_consensus": _prob_to_american(p_away_vigged),
        "n_books": len(h_probs),
    }


def ingest(json_path: Path) -> pd.DataFrame:
    logger.info(f"Loading {json_path.name}...")
    with open(json_path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data):,} dates")

    rows = []
    n_reg = n_kept = n_dropped_no_odds = 0
    for date, games in data.items():
        for g in games:
            gv = g.get("gameView", {})
            if gv.get("gameType") != "R":
                continue
            n_reg += 1
            hs, as_ = gv.get("homeTeamScore"), gv.get("awayTeamScore")
            if hs is None or as_ is None:
                continue

            parsed = parse_game(g)
            if parsed is None:
                n_dropped_no_odds += 1
                continue

            rows.append({
                "date": date,
                "season": int(date[:4]),
                "home_team_raw": gv.get("homeTeam", {}).get("shortName"),
                "away_team_raw": gv.get("awayTeam", {}).get("shortName"),
                "home_score": hs,
                "away_score": as_,
                "home_team_won": int(hs > as_),
                **parsed,
            })
            n_kept += 1

    logger.info(f"Regular-season games total: {n_reg:,}")
    logger.info(f"Kept (clean odds + results):  {n_kept:,}")
    logger.info(f"Dropped (insufficient clean books): {n_dropped_no_odds:,}")
    return pd.DataFrame(rows)


def reconcile_team_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    ESPN_TO_BBREF = {
        "SD": "SDP", "TB": "TBR", "KC": "KCR", "SF": "SFG", "WSH": "WSN",
        "CWS": "CHW", "AZ": "ARI", "ATH": "OAK",
    }
    df = df.copy()
    df["home_team"] = df["home_team_raw"].replace(ESPN_TO_BBREF)
    df["away_team"] = df["away_team_raw"].replace(ESPN_TO_BBREF)
    return df


def main():
    json_path = DATA_DIR / "raw" / "mlb_odds_dataset.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Expected {json_path}. Download the JSON from "
            f"https://github.com/ArnavSaraogi/mlb-odds-scraper/releases/tag/dataset"
        )

    df = ingest(json_path)
    df = reconcile_team_abbreviations(df)

    from sklearn.metrics import log_loss, accuracy_score
    ll = log_loss(df["home_team_won"], df["p_home_market"])
    acc = accuracy_score(df["home_team_won"], (df["p_home_market"] >= 0.5).astype(int))
    logger.info(f"Market as predictor: log_loss={ll:.4f}  acc={acc:.4f}")
    logger.info(f"Home win rate: {df['home_team_won'].mean():.3f}  "
                f"Market avg p_home: {df['p_home_market'].mean():.3f}")

    out = df[[
        "date", "season", "home_team", "away_team",
        "home_score", "away_score", "home_team_won",
        "home_ml_consensus", "away_ml_consensus",
        "p_home_market", "p_away_market", "n_books",
    ]].sort_values("date").reset_index(drop=True)

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "odds_frame.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Saved {len(out):,} games to {out_path}")


if __name__ == "__main__":
    main()