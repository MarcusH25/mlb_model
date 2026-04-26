"""
Step 8b-2: Real-market betting evaluation.

Walk-forward train on prior seasons, predict current season, compare to the
REAL closing-line market probabilities. Bet when our model disagrees with
the market by >= threshold, at the real market price. Report ROI by threshold
and confidence bucket.

Run:
    python -m src.evaluate
"""
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

from src.utils import load_config, get_logger, DATA_DIR, OUTPUT_DIR

logger = get_logger(__name__)

NON_FEATURE_COLS = {
    "season", "date", "home_team", "away_team",
    "home_score", "away_score", "home_team_won",
    "home_sp_id", "home_sp_name", "away_sp_id", "away_sp_name",
    # Odds columns we're about to join in — never features
    "home_ml_consensus", "away_ml_consensus",
    "p_home_market", "p_away_market", "n_books",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns
            if c.startswith("delta_") and c not in NON_FEATURE_COLS]


def build_model(random_seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_seed,
        tree_method="hist",
        n_jobs=-1,
    )


# ---------- Betting math ----------

def prob_to_american(p: float) -> int:
    """Fair prob -> American odds (no vig)."""
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def profit_on_win(stake: float, american_odds: int) -> float:
    """Profit on a winning bet at these odds. Handles edge cases:
    - 0: treat as even money (+100)
    - |odds| < 100: not a real American odds value, treat as even money
    """
    if american_odds == 0 or abs(american_odds) < 100:
        return stake
    if american_odds > 0:
        return stake * american_odds / 100.0
    return stake * 100.0 / abs(american_odds)

def simulate_bets(preds: pd.DataFrame, edge_threshold: float, stake: float = 100.0):
    """preds must have: home_team_won, p_home_model, p_home_market,
       home_ml_consensus, away_ml_consensus.
    Bets home when p_home_model > p_home_market + threshold (and vice versa).
    """
    p_away_model = 1 - preds["p_home_model"]
    p_away_market = 1 - preds["p_home_market"]

    bet_home = preds["p_home_model"] > preds["p_home_market"] + edge_threshold
    bet_away = p_away_model > p_away_market + edge_threshold

    bets = []
    for i, (idx, r) in enumerate(preds.iterrows()):
        if bet_home.iloc[i]:
            odds = r["home_ml_consensus"]
            won = int(r["home_team_won"] == 1)
            edge = r["p_home_model"] - r["p_home_market"]
            side = "home"
            p_model = r["p_home_model"]
            p_market = r["p_home_market"]
        elif bet_away.iloc[i]:
            odds = r["away_ml_consensus"]
            won = int(r["home_team_won"] == 0)
            edge = (1 - r["p_home_model"]) - (1 - r["p_home_market"])
            side = "away"
            p_model = 1 - r["p_home_model"]
            p_market = 1 - r["p_home_market"]
        else:
            continue

        profit = profit_on_win(stake, odds) if won else -stake
        bets.append({
            "date": r["date"], "side": side,
            "odds": odds, "won": won, "profit": profit,
            "p_model": p_model, "p_market": p_market, "edge": edge,
        })

    bets_df = pd.DataFrame(bets)
    if len(bets_df) == 0:
        return bets_df, {"n_bets": 0, "hit_rate": 0.0, "total_profit": 0.0, "roi_pct": 0.0}

    total_staked = stake * len(bets_df)
    stats = {
        "n_bets": len(bets_df),
        "hit_rate": float(bets_df["won"].mean()),
        "total_staked": total_staked,
        "total_profit": float(bets_df["profit"].sum()),
        "roi_pct": 100.0 * bets_df["profit"].sum() / total_staked,
    }
    return bets_df, stats


# ---------- Main pipeline ----------

def main():
    model_cfg = load_config("model")
    random_seed = model_cfg["project"]["random_seed"]
    target = model_cfg["data"]["target"]

    logger.info("Loading game frame and odds frame...")
    gf = pd.read_csv(DATA_DIR / "features" / "game_frame.csv", parse_dates=["date"])
    of = pd.read_csv(DATA_DIR / "processed" / "odds_frame.csv", parse_dates=["date"])
    logger.info(f"  game_frame: {len(gf):,} games")
    logger.info(f"  odds_frame: {len(of):,} games")

    # Join on (date, home_team, away_team)
    of_join = of[[
        "date", "home_team", "away_team",
        "home_ml_consensus", "away_ml_consensus",
        "p_home_market", "p_away_market", "n_books",
    ]]
    joined = gf.merge(of_join, on=["date", "home_team", "away_team"], how="inner")
    logger.info(f"After odds join (inner): {len(joined):,} games")
    logger.info(f"  Lost {len(gf) - len(joined):,} games from game_frame (no odds match)")

    feature_cols = get_feature_columns(joined)
    logger.info(f"Using {len(feature_cols)} delta features")

    # Walk-forward: predict each eval season using prior seasons as training
    seasons = sorted(joined["season"].unique())
    eval_seasons = [s for s in seasons if s >= 2023]
    logger.info(f"Training pool seasons: {seasons}. Evaluating: {eval_seasons}")

    all_preds = []
    for eval_season in eval_seasons:
        train_mask = joined["season"] < eval_season
        eval_mask = joined["season"] == eval_season
        X_tr = joined.loc[train_mask, feature_cols]
        y_tr = joined.loc[train_mask, target]
        X_ev = joined.loc[eval_mask, feature_cols]
        meta_cols = ["date", "season", "home_team", "away_team", "home_team_won",
                     "home_ml_consensus", "away_ml_consensus",
                     "p_home_market", "p_away_market"]
        meta_ev = joined.loc[eval_mask, meta_cols].copy()

        logger.info(f"Eval {eval_season}: train={len(X_tr):,} eval={len(X_ev):,}")
        model = build_model(random_seed)
        model.fit(X_tr, y_tr)
        meta_ev["p_home_model"] = model.predict_proba(X_ev)[:, 1]
        all_preds.append(meta_ev)

    preds = pd.concat(all_preds, ignore_index=True)

    # ==== Model vs Market: head-to-head on same games ====
    model_ll = log_loss(preds["home_team_won"], preds["p_home_model"])
    market_ll = log_loss(preds["home_team_won"], preds["p_home_market"])
    model_acc = accuracy_score(preds["home_team_won"], (preds["p_home_model"] >= 0.5).astype(int))
    market_acc = accuracy_score(preds["home_team_won"], (preds["p_home_market"] >= 0.5).astype(int))

    logger.info("=" * 70)
    logger.info(f"HEAD-TO-HEAD ON {len(preds):,} OUT-OF-SAMPLE GAMES ({eval_seasons})")
    logger.info("=" * 70)
    logger.info(f"Our XGBoost:    log_loss={model_ll:.4f}  acc={model_acc:.4f}")
    logger.info(f"Closing market: log_loss={market_ll:.4f}  acc={market_acc:.4f}")
    logger.info(f"Gap (we want model lower): {model_ll - market_ll:+.4f}")

    # ==== ROI by edge threshold (the money shot) ====
    logger.info("\n=== ROI vs Edge Threshold (flat $100 bets, real market prices) ===")
    logger.info(f"{'Edge':>6} | {'# bets':>7} | {'Hit%':>6} | {'Profit':>10} | {'ROI%':>7}")
    logger.info("-" * 60)
    roi_results = []
    for thresh in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        _, stats = simulate_bets(preds, edge_threshold=thresh)
        roi_results.append({"edge": thresh, **stats})
        logger.info(
            f"{thresh:>5.2f}  | {stats['n_bets']:>7d} | "
            f"{100*stats['hit_rate']:>5.1f}% | "
            f"{stats['total_profit']:>+10.2f} | "
            f"{stats['roi_pct']:>+6.2f}%"
        )

    # ==== Per-season ROI at modest edge threshold ====
    logger.info(f"\n=== Per-season breakdown @ edge=0.03 ===")
    logger.info(f"{'Season':>7} | {'# bets':>7} | {'Hit%':>6} | {'ROI%':>7}")
    logger.info("-" * 40)
    for s in eval_seasons:
        season_preds = preds[preds["season"] == s]
        _, stats = simulate_bets(season_preds, edge_threshold=0.03)
        logger.info(
            f"{s:>7d} | {stats['n_bets']:>7d} | "
            f"{100*stats['hit_rate']:>5.1f}% | "
            f"{stats['roi_pct']:>+6.2f}%"
        )

    # ==== Where we disagree with the market ====
    logger.info("\n=== Disagreement profile ===")
    preds["disagreement"] = preds["p_home_model"] - preds["p_home_market"]
    logger.info(f"Mean disagreement: {preds['disagreement'].mean():+.4f}")
    logger.info(f"Std disagreement:  {preds['disagreement'].std():.4f}")
    logger.info(f"Games where we disagree by > 0.05 (either direction): "
                f"{(preds['disagreement'].abs() > 0.05).sum():,} "
                f"({100*(preds['disagreement'].abs() > 0.05).mean():.1f}%)")
    logger.info(f"Games where we disagree by > 0.10: "
                f"{(preds['disagreement'].abs() > 0.10).sum():,} "
                f"({100*(preds['disagreement'].abs() > 0.10).mean():.1f}%)")

    # Persist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUTPUT_DIR / "walk_forward_predictions_real_odds.csv", index=False)
    pd.DataFrame(roi_results).to_csv(OUTPUT_DIR / "real_roi_by_edge.csv", index=False)
    logger.info(f"\nSaved predictions and ROI table to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()