"""
Step 9b: Out-of-sample niche test on 2025.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score

from src.utils import load_config, get_logger, DATA_DIR, OUTPUT_DIR

logger = get_logger(__name__)

NON_FEATURE_COLS = {
    "season", "date", "home_team", "away_team",
    "home_score", "away_score", "home_team_won",
    "home_sp_id", "home_sp_name", "away_sp_id", "away_sp_name",
    "home_ml_consensus", "away_ml_consensus",
    "p_home_market", "p_away_market", "n_books",
}

EXTRA_FEATURES = [
    "home_park_factor",
]

EDGE_THRESHOLD = 0.02
STAKE = 100.0


def profit_on_win(stake, american_odds):
    if american_odds == 0 or abs(american_odds) < 100:
        return stake
    if american_odds > 0:
        return stake * american_odds / 100.0
    return stake * 100.0 / abs(american_odds)


def get_feature_columns(df):
    candidates = [c for c in df.select_dtypes(include=[np.number]).columns
                  if c not in NON_FEATURE_COLS]
    return [c for c in candidates
            if c.startswith("delta_") or c in EXTRA_FEATURES]


def build_model(random_seed):
    return XGBClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=3.0,
        objective="binary:logistic", eval_metric="logloss",
        random_state=random_seed, tree_method="hist", n_jobs=-1,
    )


def main():
    model_cfg = load_config("model")
    random_seed = model_cfg["project"]["random_seed"]
    target = model_cfg["data"]["target"]

    logger.info("Loading game frame and odds frame...")
    gf = pd.read_csv(DATA_DIR / "features" / "game_frame.csv", parse_dates=["date"])
    of = pd.read_csv(DATA_DIR / "processed" / "odds_frame.csv", parse_dates=["date"])

    of_join = of[[
        "date", "home_team", "away_team",
        "home_ml_consensus", "away_ml_consensus",
        "p_home_market", "p_away_market", "n_books",
    ]]
    joined = gf.merge(of_join, on=["date", "home_team", "away_team"], how="inner")
    logger.info(f"After odds join: {len(joined):,} games")

    train_df = joined[joined["season"] < 2025].copy()
    test_df = joined[joined["season"] == 2025].copy()
    logger.info(f"Train (2021-2024): {len(train_df):,} games")
    logger.info(f"Test (2025):       {len(test_df):,} games")

    if len(test_df) == 0:
        logger.error("No 2025 games found!")
        return

    feature_cols = get_feature_columns(joined)
    logger.info(f"Using {len(feature_cols)} features")
    extras_in_use = [c for c in feature_cols if c in EXTRA_FEATURES]
    logger.info(f"Extra features active: {extras_in_use}")

    X_tr, y_tr = train_df[feature_cols], train_df[target]
    X_te = test_df[feature_cols]

    logger.info("Training XGBoost on 2021-2024...")
    model = build_model(random_seed)
    model.fit(X_tr, y_tr)
    test_df["p_home_model"] = model.predict_proba(X_te)[:, 1]

    model_ll = log_loss(test_df[target], test_df["p_home_model"])
    market_ll = log_loss(test_df[target], test_df["p_home_market"])
    model_acc = accuracy_score(test_df[target], (test_df["p_home_model"] >= 0.5).astype(int))
    market_acc = accuracy_score(test_df[target], (test_df["p_home_market"] >= 0.5).astype(int))
    logger.info("\n" + "=" * 70)
    logger.info(f"OUT-OF-SAMPLE ON 2025 ({len(test_df):,} games)")
    logger.info("=" * 70)
    logger.info(f"Our XGBoost:    log_loss={model_ll:.4f}  acc={model_acc:.4f}")
    logger.info(f"Closing market: log_loss={market_ll:.4f}  acc={market_acc:.4f}")
    logger.info(f"Gap: {model_ll - market_ll:+.4f}")
    logger.info(f"\nBaseline (delta-only, no extras): log_loss=0.6839  acc=0.5604")

    p_away_model = 1 - test_df["p_home_model"]
    p_away_market = 1 - test_df["p_home_market"]
    niche_mask = (p_away_model > p_away_market + EDGE_THRESHOLD) & (test_df["away_ml_consensus"] > 0)
    niche_bets = test_df[niche_mask].copy()
    niche_bets["won"] = (niche_bets["home_team_won"] == 0).astype(int)
    niche_bets["profit"] = niche_bets.apply(
        lambda r: profit_on_win(STAKE, int(r["away_ml_consensus"])) if r["won"] else -STAKE,
        axis=1
    )

    logger.info("\n" + "=" * 70)
    logger.info(f"NICHE TEST (away underdog + edge >= {EDGE_THRESHOLD})")
    logger.info("=" * 70)
    logger.info(f"Niche bets placed: {len(niche_bets):,}")

    if len(niche_bets) == 0:
        logger.warning("No bets qualified.")
        return

    hit_rate = niche_bets["won"].mean()
    total_profit = niche_bets["profit"].sum()
    roi_pct = 100.0 * total_profit / (STAKE * len(niche_bets))
    avg_odds = niche_bets["away_ml_consensus"].mean()

    logger.info(f"Avg odds taken: +{avg_odds:.0f}")
    logger.info(f"Hit rate:       {hit_rate:.3f}")
    logger.info(f"Total profit:   ${total_profit:+,.2f}")
    logger.info(f"ROI:            {roi_pct:+.2f}%")
    logger.info(f"\nBaseline niche ROI (delta-only): -0.64%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    niche_bets.to_csv(OUTPUT_DIR / "niche_test_2025_bets.csv", index=False)


if __name__ == "__main__":
    main()