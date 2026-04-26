"""
Step 9: Niche analysis — find pockets where our model beats the market.

Uses walk-forward predictions from evaluate.py. For each subset of games, reports
per-season ROI at a fixed edge threshold. Flags slices that are positive in BOTH
eval seasons (2023 AND 2024) as candidate real niches.

Run:
    python -m src.niche_analysis
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils import get_logger, OUTPUT_DIR

logger = get_logger(__name__)

STAKE = 100.0
EDGE_THRESHOLD = 0.02  # matches our "least bad" overall threshold


def profit_on_win(stake: float, american_odds: int) -> float:
    if american_odds == 0 or abs(american_odds) < 100:
        return stake
    if american_odds > 0:
        return stake * american_odds / 100.0
    return stake * 100.0 / abs(american_odds)


def add_bet_outcome(preds: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    """Add 'side', 'odds', 'won', 'profit' columns. Rows with no bet = NaN."""
    preds = preds.copy()
    p_away_model = 1 - preds["p_home_model"]
    p_away_market = 1 - preds["p_home_market"]

    bet_home = preds["p_home_model"] > preds["p_home_market"] + edge_threshold
    bet_away = p_away_model > p_away_market + edge_threshold

    preds["side"] = np.where(bet_home, "home", np.where(bet_away, "away", None))
    preds["odds"] = np.where(
        bet_home, preds["home_ml_consensus"],
        np.where(bet_away, preds["away_ml_consensus"], np.nan)
    )
    preds["won"] = np.where(
        bet_home, (preds["home_team_won"] == 1).astype(int),
        np.where(bet_away, (preds["home_team_won"] == 0).astype(int), np.nan)
    )

    def compute_profit(r):
        if r["side"] is None or pd.isna(r["side"]):
            return np.nan
        return profit_on_win(STAKE, int(r["odds"])) if r["won"] == 1 else -STAKE

    preds["profit"] = preds.apply(compute_profit, axis=1)
    preds["bet_placed"] = preds["side"].notna()
    return preds


def roi_for_slice(slice_df: pd.DataFrame) -> dict:
    """Compute ROI stats for a slice of bet-placed rows only."""
    bets = slice_df[slice_df["bet_placed"]]
    if len(bets) == 0:
        return {"n_bets": 0, "hit_rate": np.nan, "roi_pct": np.nan, "total_profit": 0.0}
    return {
        "n_bets": len(bets),
        "hit_rate": float(bets["won"].mean()),
        "total_profit": float(bets["profit"].sum()),
        "roi_pct": 100.0 * bets["profit"].sum() / (STAKE * len(bets)),
    }


def two_season_slice(preds: pd.DataFrame, mask_func, label: str) -> dict:
    """Run a slice across 2023 and 2024 separately, plus combined."""
    results = {"label": label}
    for season in [2023, 2024]:
        season_preds = preds[preds["season"] == season]
        mask = mask_func(season_preds)
        sliced = season_preds[mask]
        stats = roi_for_slice(sliced)
        results[f"{season}_n"] = stats["n_bets"]
        results[f"{season}_hit"] = stats["hit_rate"]
        results[f"{season}_roi"] = stats["roi_pct"]

    # Combined
    mask = mask_func(preds)
    combined = preds[mask]
    stats = roi_for_slice(combined)
    results["combined_n"] = stats["n_bets"]
    results["combined_hit"] = stats["hit_rate"]
    results["combined_roi"] = stats["roi_pct"]
    return results


def main():
    preds_path = OUTPUT_DIR / "walk_forward_predictions_real_odds.csv"
    preds = pd.read_csv(preds_path, parse_dates=["date"])
    logger.info(f"Loaded {len(preds):,} predictions")

    preds = add_bet_outcome(preds, EDGE_THRESHOLD)
    logger.info(f"Bets placed at edge={EDGE_THRESHOLD}: {preds['bet_placed'].sum():,}")

    # Derived columns for slicing
    preds["month"] = preds["date"].dt.month
    preds["day_of_week"] = preds["date"].dt.day_name()
    preds["home_is_fav"] = preds["p_home_market"] > 0.5
    preds["home_underdog_size"] = np.where(
        preds["home_ml_consensus"] > 0, preds["home_ml_consensus"], 0
    )
    preds["away_underdog_size"] = np.where(
        preds["away_ml_consensus"] > 0, preds["away_ml_consensus"], 0
    )
    preds["disagreement"] = preds["p_home_model"] - preds["p_home_market"]

    # ============ SLICES ============
    slices = []

    # --- By bet side ---
    slices.append(two_season_slice(preds, lambda d: d["side"] == "home", "Bet home only"))
    slices.append(two_season_slice(preds, lambda d: d["side"] == "away", "Bet away only"))

    # --- By underdog status ---
    slices.append(two_season_slice(
        preds, lambda d: (d["side"] == "home") & (d["home_ml_consensus"] > 0),
        "Bet home when home is underdog"
    ))
    slices.append(two_season_slice(
        preds, lambda d: (d["side"] == "away") & (d["away_ml_consensus"] > 0),
        "Bet away when away is underdog"
    ))
    slices.append(two_season_slice(
        preds, lambda d: (d["side"] == "home") & (d["home_ml_consensus"] < 0),
        "Bet home when home is favorite"
    ))
    slices.append(two_season_slice(
        preds, lambda d: (d["side"] == "away") & (d["away_ml_consensus"] < 0),
        "Bet away when away is favorite"
    ))

    # --- Big underdogs (juicy payouts) ---
    for thresh in [120, 140, 160, 180, 200]:
        slices.append(two_season_slice(
            preds, lambda d, t=thresh: (
                ((d["side"] == "home") & (d["home_ml_consensus"] >= t)) |
                ((d["side"] == "away") & (d["away_ml_consensus"] >= t))
            ),
            f"Underdog bets at +{thresh} or bigger",
        ))

    # --- By month ---
    for m in sorted(preds["month"].dropna().unique()):
        if pd.isna(m):
            continue
        slices.append(two_season_slice(
            preds, lambda d, mm=int(m): (d["bet_placed"]) & (d["month"] == mm),
            f"Month {int(m):02d}",
        ))

    # --- By disagreement size (confidence bucket) ---
    slices.append(two_season_slice(
        preds, lambda d: d["bet_placed"] & (d["disagreement"].abs() < 0.03),
        "Mild disagreement (|edge| < 0.03)",
    ))
    slices.append(two_season_slice(
        preds, lambda d: d["bet_placed"] & (d["disagreement"].abs() >= 0.03) & (d["disagreement"].abs() < 0.07),
        "Moderate disagreement (|edge| 0.03-0.07)",
    ))
    slices.append(two_season_slice(
        preds, lambda d: d["bet_placed"] & (d["disagreement"].abs() >= 0.07),
        "Large disagreement (|edge| >= 0.07)",
    ))

    # --- Cross: underdog AND moderate edge ---
    slices.append(two_season_slice(
        preds, lambda d: (
            d["bet_placed"] &
            (d["disagreement"].abs() >= 0.03) &
            (d["disagreement"].abs() < 0.07) &
            (
                ((d["side"] == "home") & (d["home_ml_consensus"] > 0)) |
                ((d["side"] == "away") & (d["away_ml_consensus"] > 0))
            )
        ),
        "Moderate edge + underdog bet",
    ))

    # ============ REPORT ============
    df = pd.DataFrame(slices)

    # Show all slices
    logger.info("\n" + "=" * 110)
    logger.info("ALL SLICES (2023 and 2024 separately)")
    logger.info("=" * 110)
    logger.info(f"{'Slice':<45} | {'2023 n':>7} {'2023 ROI':>9} | {'2024 n':>7} {'2024 ROI':>9} | {'Combined':>8} {'ROI':>7}")
    logger.info("-" * 110)
    for _, r in df.iterrows():
        if r["combined_n"] < 30:  # too small to mention
            continue
        roi_23 = f"{r['2023_roi']:+6.2f}%" if pd.notna(r['2023_roi']) else "   n/a"
        roi_24 = f"{r['2024_roi']:+6.2f}%" if pd.notna(r['2024_roi']) else "   n/a"
        roi_c = f"{r['combined_roi']:+6.2f}%" if pd.notna(r['combined_roi']) else "   n/a"
        logger.info(
            f"{r['label']:<45} | {int(r['2023_n']):>7d} {roi_23:>9} | "
            f"{int(r['2024_n']):>7d} {roi_24:>9} | {int(r['combined_n']):>8d} {roi_c:>7}"
        )

    # Promising: positive in BOTH seasons AND combined AND >= 100 bets
    logger.info("\n" + "=" * 110)
    logger.info("PROMISING SLICES (positive in both seasons, >=100 combined bets)")
    logger.info("=" * 110)
    promising = df[
        (df["2023_roi"] > 0) & (df["2024_roi"] > 0) &
        (df["combined_roi"] > 0) & (df["combined_n"] >= 100)
    ]
    if len(promising) == 0:
        logger.info("None. Model has no consistent profitable niche in 2023 + 2024.")
    else:
        for _, r in promising.iterrows():
            logger.info(
                f"  {r['label']:<45} 2023: {r['2023_roi']:+.2f}% ({int(r['2023_n'])}) | "
                f"2024: {r['2024_roi']:+.2f}% ({int(r['2024_n'])}) | "
                f"combined: {r['combined_roi']:+.2f}% ({int(r['combined_n'])})"
            )

    df.to_csv(OUTPUT_DIR / "niche_analysis.csv", index=False)
    logger.info(f"\nSaved full slice table to {OUTPUT_DIR}/niche_analysis.csv")


if __name__ == "__main__":
    main()