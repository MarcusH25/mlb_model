"""
Step 7: XGBoost model (tuned).

Uses a small regularized model with early stopping to fight overfitting
on our ~8K-row, mostly-correlated-features dataset.

Run:
    python -m src.train_xgb
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score
import joblib

from src.utils import load_config, get_logger, DATA_DIR, MODELS_DIR, OUTPUT_DIR

logger = get_logger(__name__)

NON_FEATURE_COLS = {
    "season", "date", "home_team", "away_team",
    "home_score", "away_score", "home_team_won",
    "home_sp_id", "home_sp_name", "away_sp_id", "away_sp_name",
}


def get_feature_columns(df: pd.DataFrame, deltas_only: bool = True) -> list[str]:
    candidates = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numeric = df[candidates].select_dtypes(include=[np.number]).columns.tolist()
    if deltas_only:
        numeric = [c for c in numeric if c.startswith("delta_")]
    return numeric


def build_model(random_seed: int) -> XGBClassifier:
    """Small, heavily regularized model with early stopping.

    Key changes from prior version:
      - max_depth 3 (was 4) — shallower trees
      - n_estimators 800 but early_stopping_rounds=30 (was 400, no stopping)
      - reg_lambda 3.0 (was 1.0) — stronger L2
      - learning_rate 0.03 (was 0.05) — smaller steps
      - min_child_weight 10 (was 5) — each leaf needs more samples
    """
    return XGBClassifier(
        n_estimators=800,
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
        early_stopping_rounds=30,
    )


def time_series_cv(X: pd.DataFrame, y: pd.Series, n_splits: int, random_seed: int) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Carve a small validation set from the tail of training for early stopping
        val_frac = 0.15
        val_cut = int(len(X_tr) * (1 - val_frac))
        X_fit, X_val = X_tr.iloc[:val_cut], X_tr.iloc[val_cut:]
        y_fit, y_val = y_tr.iloc[:val_cut], y_tr.iloc[val_cut:]

        model = build_model(random_seed)
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        m = {
            "fold": fold,
            "train_n": len(train_idx),
            "test_n": len(test_idx),
            "best_iter": int(model.best_iteration),
            "log_loss": float(log_loss(y_te, proba)),
            "brier": float(brier_score_loss(y_te, proba)),
            "accuracy": float(accuracy_score(y_te, pred)),
            "auc": float(roc_auc_score(y_te, proba)),
        }
        fold_metrics.append(m)
        logger.info(
            f"Fold {fold}: train={m['train_n']:>5d} test={m['test_n']:>5d} "
            f"iters={m['best_iter']:>3d} | "
            f"log_loss={m['log_loss']:.4f} brier={m['brier']:.4f} "
            f"acc={m['accuracy']:.4f} auc={m['auc']:.4f}"
        )

    summary = {
        "n_folds": n_splits,
        "mean_log_loss": float(np.mean([m["log_loss"] for m in fold_metrics])),
        "mean_brier": float(np.mean([m["brier"] for m in fold_metrics])),
        "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "mean_auc": float(np.mean([m["auc"] for m in fold_metrics])),
        "folds": fold_metrics,
    }
    return summary


def report_feature_importance(model: XGBClassifier, feature_names: list[str], top_n: int = 20):
    imp = pd.DataFrame({"feature": feature_names, "gain": model.feature_importances_}).sort_values("gain", ascending=False)
    logger.info(f"Top {top_n} features by importance:")
    for _, r in imp.head(top_n).iterrows():
        logger.info(f"  {r['feature']:<45} {r['gain']:.4f}")
    return imp


def main():
    model_cfg = load_config("model")
    random_seed = model_cfg["project"]["random_seed"]
    target = model_cfg["data"]["target"]
    cv_splits = model_cfg["model"]["cv_splits"]

    logger.info("Loading game frame...")
    gf = pd.read_csv(DATA_DIR / "features" / "game_frame.csv", parse_dates=["date"])
    gf = gf.sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(gf):,} games")

    # Using deltas only — same feature set that worked for logreg, fair comparison
    feature_cols = get_feature_columns(gf, deltas_only=True)
    logger.info(f"Using {len(feature_cols)} features (deltas only)")

    X = gf[feature_cols]
    y = gf[target]

    logger.info(f"Running {cv_splits}-fold time-series CV with early stopping...")
    cv_summary = time_series_cv(X, y, cv_splits, random_seed)

    logger.info("=" * 70)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 70)
    logger.info(f"XGBoost (tuned):               log_loss={cv_summary['mean_log_loss']:.4f} acc={cv_summary['mean_accuracy']:.4f} auc={cv_summary['mean_auc']:.4f}")
    logger.info(f"Logreg deltas-only benchmark:  log_loss=0.6856        acc=0.5607 auc=0.5820")
    logger.info(f"Always-home benchmark:         log_loss=0.6911        acc=0.5320 auc=0.5000")
    logger.info("=" * 70)

    logger.info("Training final model on full dataset...")
    # For the final model, use the median best_iter across folds as n_estimators
    median_iter = int(np.median([m["best_iter"] for m in cv_summary["folds"]])) + 1
    logger.info(f"Using {median_iter} trees (median best_iter from CV)")

    final_model = XGBClassifier(
        n_estimators=median_iter,
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
    final_model.fit(X, y)
    imp = report_feature_importance(final_model, feature_cols)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": final_model, "feature_cols": feature_cols},
        MODELS_DIR / "xgb_baseline.pkl",
    )
    with open(OUTPUT_DIR / "xgb_baseline_metrics.json", "w") as f:
        json.dump(cv_summary, f, indent=2)
    imp.to_csv(OUTPUT_DIR / "xgb_feature_importance.csv", index=False)
    logger.info("Saved model, metrics, and feature importance")


if __name__ == "__main__":
    main()