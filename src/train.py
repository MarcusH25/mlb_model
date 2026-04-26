"""
Step 6: Baseline model training.

Trains a logistic regression on the game frame using time-series cross-validation.
Reports log loss, Brier score, accuracy, and saves the trained model + CV metrics.

Supports two feature sets via CLI:
  python -m src.train                 # all features (original)
  python -m src.train --deltas-only   # only delta_* features (diagnostic)
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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


def get_feature_columns(df: pd.DataFrame, deltas_only: bool) -> list[str]:
    candidates = [c for c in df.columns if c not in NON_FEATURE_COLS]
    numeric = df[candidates].select_dtypes(include=[np.number]).columns.tolist()
    if deltas_only:
        numeric = [c for c in numeric if c.startswith("delta_")]
    return numeric


def build_pipeline(random_seed: int, C: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            max_iter=2000,
            solver="lbfgs",
            random_state=random_seed,
        )),
    ])


def time_series_cv(X: pd.DataFrame, y: pd.Series, n_splits: int, random_seed: int, C: float) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe = build_pipeline(random_seed, C)
        pipe.fit(X_tr, y_tr)

        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        m = {
            "fold": fold,
            "train_n": len(train_idx),
            "test_n": len(test_idx),
            "log_loss": float(log_loss(y_te, proba)),
            "brier": float(brier_score_loss(y_te, proba)),
            "accuracy": float(accuracy_score(y_te, pred)),
            "auc": float(roc_auc_score(y_te, proba)),
            "home_win_rate_test": float(y_te.mean()),
        }
        fold_metrics.append(m)
        logger.info(
            f"Fold {fold}: train={m['train_n']:>5d} test={m['test_n']:>5d} | "
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


def train_final_model(X: pd.DataFrame, y: pd.Series, random_seed: int, C: float) -> Pipeline:
    pipe = build_pipeline(random_seed, C)
    pipe.fit(X, y)
    return pipe


def report_top_coefficients(pipe: Pipeline, feature_names: list[str], top_n: int = 15):
    model = pipe.named_steps["model"]
    coefs = pd.DataFrame({"feature": feature_names, "coef": model.coef_[0]}).sort_values("coef")

    logger.info(f"Top {top_n} NEGATIVE coefficients (push prediction toward AWAY win):")
    for _, r in coefs.head(top_n).iterrows():
        logger.info(f"  {r['feature']:<45} {r['coef']:+.4f}")
    logger.info(f"Top {top_n} POSITIVE coefficients (push prediction toward HOME win):")
    for _, r in coefs.tail(top_n).iterrows():
        logger.info(f"  {r['feature']:<45} {r['coef']:+.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deltas-only", action="store_true", help="Use only delta_* features")
    p.add_argument("--C", type=float, default=None, help="L2 regularization (lower = stronger)")
    return p.parse_args()


def main():
    args = parse_args()
    model_cfg = load_config("model")
    random_seed = model_cfg["project"]["random_seed"]
    target = model_cfg["data"]["target"]
    cv_splits = model_cfg["model"]["cv_splits"]

    # Stronger regularization for deltas-only is still useful; default to 0.1
    C = args.C if args.C is not None else (0.1 if args.deltas_only else 1.0)
    tag = "deltas_only" if args.deltas_only else "all_features"

    logger.info(f"=== Config: {tag} | C={C} ===")
    logger.info("Loading game frame...")
    gf = pd.read_csv(DATA_DIR / "features" / "game_frame.csv", parse_dates=["date"])
    gf = gf.sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(gf):,} games")

    feature_cols = get_feature_columns(gf, args.deltas_only)
    logger.info(f"Using {len(feature_cols)} features ({tag})")

    X = gf[feature_cols]
    y = gf[target]

    logger.info(f"Target balance: {y.mean():.3f} (home win rate)")

    logger.info(f"Running {cv_splits}-fold time-series CV...")
    cv_summary = time_series_cv(X, y, cv_splits, random_seed, C)
    logger.info(
        f"CV MEANS: log_loss={cv_summary['mean_log_loss']:.4f} "
        f"brier={cv_summary['mean_brier']:.4f} "
        f"acc={cv_summary['mean_accuracy']:.4f} "
        f"auc={cv_summary['mean_auc']:.4f}"
    )

    baseline_log_loss = log_loss(y, [y.mean()] * len(y))
    logger.info(f"Benchmark (constant home-win-rate): log_loss={baseline_log_loss:.4f}")

    logger.info("Training final model on full dataset...")
    final_pipe = train_final_model(X, y, random_seed, C)
    report_top_coefficients(final_pipe, feature_cols)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"logreg_{tag}.pkl"
    joblib.dump({"pipeline": final_pipe, "feature_cols": feature_cols}, model_path)
    logger.info(f"Saved model to {model_path}")

    metrics_path = OUTPUT_DIR / f"logreg_{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()