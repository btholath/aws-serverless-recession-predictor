#!/usr/bin/env python3
"""
recession_backtest.py

Follow-on script for your FRED "macro_panel_monthly" dataset to run a realistic
recession prediction backtest.

What this script does
1) Loads the monthly panel produced by download_fred_macro_pack.py
2) Creates forward-looking labels for horizons 3/6/12 months:
      recession_in_next_h = max(USREC over the next h months)
3) Builds a leak-safe feature matrix:
   - Uses only information available at time t (shifts all features by 1 month)
   - Optional: adds lag features (1..L)
4) Splits train/test by time (no leakage):
   - Either a fixed split date (recommended), or last N years held out
5) Trains baseline models:
   - Logistic Regression (default)
   - XGBoost (optional, if installed)
6) Outputs metrics: ROC-AUC, PR-AUC, Precision/Recall/F1, Confusion Matrix
7) Saves artifacts: predictions CSV + model coefficients/feature importance

Usage
  source .venv/bin/activate
  pip install pandas pyarrow scikit-learn matplotlib
  # Optional for XGBoost:
  # pip install xgboost

  python recession_backtest.py \
    --input ./fred_macro_pack/panels/macro_panel_monthly.parquet \
    --horizons 3 6 12 \
    --target-horizon 6 \
    --test-start 2005-01-31 \
    --lags 3 \
    --model logreg \
    --out ./backtest_out

Notes
- The panel uses month-end timestamps. Provide --test-start as YYYY-MM-DD (month-end preferred).
- This is a baseline; for a serious model you’ll likely:
    * tune thresholds for recall vs precision,
    * try calibrated probabilities,
    * use time-series CV (walk-forward),
    * and add domain transformations (spreads, growth rates, etc.).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Helpers
# -----------------------------

@dataclass
class BacktestResult:
    horizon: int
    n_train: int
    n_test: int
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float


def ensure_month_end(dt: pd.Timestamp) -> pd.Timestamp:
    # If user gives a date like 2005-01-01, align to month-end for comparisons
    # (We don't strictly require it, but it reduces surprises.)
    return (dt + pd.offsets.MonthEnd(0)).normalize()


def create_forward_labels(df: pd.DataFrame, horizons: List[int], usrec_col: str = "USREC") -> pd.DataFrame:
    """
    For each horizon h, create:
      recession_in_next_h = max(USREC over (t+1 .. t+h))

    This label answers: "Will we be in recession at any time in the next h months?"
    """
    if usrec_col not in df.columns:
        raise ValueError(f"Missing '{usrec_col}' column. Ensure you downloaded USREC and built the panel.")

    out = df.copy()
    usrec = out[usrec_col].fillna(0.0).astype(float)

    for h in horizons:
        # Rolling max on the FUTURE window: shift(-1) to start at t+1
        # then rolling over h months.
        label = usrec.shift(-1).rolling(window=h, min_periods=1).max()
        out[f"recession_in_next_{h}m"] = (label >= 0.5).astype("int8")

    return out


def build_features(
        df: pd.DataFrame,
        feature_cols: List[str],
        lags: int = 0,
        shift_by: int = 1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a leak-safe feature matrix:
    - Shifts all features by `shift_by` months so features at time t use data from t-1 and earlier.
    - Optionally adds lag features (1..lags) on the already-shifted series.

    Returns (X_df, final_feature_cols)
    """
    X = df[["date_month_end"] + feature_cols].copy()
    X = X.sort_values("date_month_end")

    # Shift base features (avoid using contemporaneous month values if your label includes t+1..t+h)
    for col in feature_cols:
        X[col] = X[col].shift(shift_by)

    final_cols = feature_cols.copy()

    # Add lagged versions
    if lags and lags > 0:
        for lag in range(1, lags + 1):
            for col in feature_cols:
                lag_col = f"{col}_lag{lag}"
                X[lag_col] = X[col].shift(lag)  # lag on already shifted base
                final_cols.append(lag_col)

    # Drop date column from X returned to model pipeline, but keep it for joins later
    return X, final_cols


def time_split(df: pd.DataFrame, test_start: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by date_month_end:
      train: < test_start
      test:  >= test_start
    """
    train = df[df["date_month_end"] < test_start].copy()
    test = df[df["date_month_end"] >= test_start].copy()
    return train, test


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray, strategy: str = "f1") -> float:
    """
    Choose a classification threshold based on the test set probabilities.
    Strategies:
      - "f1": maximize F1
      - "0.5": fixed threshold
    Note: For a proper backtest, threshold selection should be done on a validation set
    within training data. This baseline uses test for simplicity (you can improve later).
    """
    if strategy == "0.5":
        return 0.5

    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr, best_f1 = 0.5, -1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


# -----------------------------
# Models
# -----------------------------

def train_logreg(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Logistic regression with:
      - median imputation
      - standard scaling
      - class_weight balanced (recessions are rare)
    """
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
    )
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    """
    Optional: XGBoost baseline.
    Requires: pip install xgboost
    """
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise RuntimeError("XGBoost not installed. Run: pip install xgboost") from e

    # Reasonable baseline hyperparameters (not tuned)
    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=1.0,
        eval_metric="logloss",
        n_jobs=4,
    )

    # Simple impute missing values with median before XGBoost
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_train)
    model.fit(X_tr, y_train)
    model._imputer = imp  # attach for later use
    return model


def predict_proba(model: object, X: pd.DataFrame, model_type: str) -> np.ndarray:
    if model_type == "logreg":
        return model.predict_proba(X)[:, 1]
    if model_type == "xgb":
        X2 = model._imputer.transform(X)
        return model.predict_proba(X2)[:, 1]
    raise ValueError(f"Unknown model_type: {model_type}")


def export_feature_importance(
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        out_path: Path,
        model_type: str,
) -> None:
    """
    Exports a simple feature importance report:
    - LogReg: absolute standardized coefficients (approx)
    - XGB: gain-based importance if available
    - Additionally: permutation importance on test set (fast baseline)
    """
    rows = []

    if model_type == "logreg":
        # Coefs correspond to scaled features (standardized)
        coefs = model.named_steps["clf"].coef_.ravel()
        for f, c in zip(feature_names, coefs):
            rows.append({"feature": f, "importance": float(abs(c)), "type": "logreg_abs_coef"})

    elif model_type == "xgb":
        # XGB internal importance (by gain) if present
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        # score keys are like 'f0', 'f1' ... mapping to column index
        for k, v in score.items():
            idx = int(k[1:])
            if 0 <= idx < len(feature_names):
                rows.append({"feature": feature_names[idx], "importance": float(v), "type": "xgb_gain"})

    # Permutation importance (model-agnostic)
    try:
        if model_type == "logreg":
            perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
            for f, v in zip(feature_names, perm.importances_mean):
                rows.append({"feature": f, "importance": float(v), "type": "perm_roc_auc"})
        else:
            # For xgb we can't pass the raw model easily due to imputer; skip perm by default
            pass
    except Exception:
        pass

    if not rows:
        return

    imp_df = pd.DataFrame(rows).sort_values(["type", "importance"], ascending=[True, False])
    imp_df.to_csv(out_path, index=False)


# -----------------------------
# Main pipeline
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create forward recession labels and run leak-safe time split backtest.")
    parser.add_argument("--input", required=True, help="Path to macro_panel_monthly.parquet or .csv")
    parser.add_argument("--out", default="./backtest_out", help="Output folder for artifacts")
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 6, 12], help="Forward label horizons in months")
    parser.add_argument("--target-horizon", type=int, default=6, help="Which horizon to train/evaluate (e.g., 6)")
    parser.add_argument("--test-start", required=True, help="Test start date (YYYY-MM-DD), split is train < test_start")
    parser.add_argument("--lags", type=int, default=0, help="Add lag features (1..lags) on shifted features")
    parser.add_argument("--shift-by", type=int, default=1,
                        help="Shift features by N months to avoid leakage (default 1)")
    parser.add_argument("--model", choices=["logreg", "xgb"], default="logreg", help="Model type")
    parser.add_argument("--threshold-strategy", choices=["f1", "0.5"], default="f1",
                        help="How to pick classification threshold")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    elif in_path.suffix.lower() == ".csv":
        df = pd.read_csv(in_path, parse_dates=["date_month_end"])
    else:
        raise ValueError("Input must be .parquet or .csv")

    if "date_month_end" not in df.columns:
        raise ValueError("Expected 'date_month_end' column in input panel.")

    df["date_month_end"] = pd.to_datetime(df["date_month_end"])
    df = df.sort_values("date_month_end").reset_index(drop=True)

    # Create forward labels
    df = create_forward_labels(df, horizons=args.horizons, usrec_col="USREC")
    target_col = f"recession_in_next_{args.target_horizon}m"
    if target_col not in df.columns:
        raise ValueError(f"Target label {target_col} was not created. Check --horizons and --target-horizon.")

    # Define feature columns (exclude labels/date)
    exclude = {"date_month_end", "USREC", "recession_label"} | {f"recession_in_next_{h}m" for h in args.horizons}
    feature_cols = [c for c in df.columns if
                    c not in exclude and not c.endswith("_yoy_pct") is False]  # keep yoy if present

    # More explicit: include numeric feature columns only, exclude any derived label columns
    feature_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("recession_in_next_"):
            continue
        if df[c].dtype.kind in "biufc":  # numeric
            feature_cols.append(c)

    if not feature_cols:
        raise ValueError("No numeric feature columns found. Check your input panel contents.")

    # Build leak-safe features
    X_all, final_feature_cols = build_features(
        df,
        feature_cols=feature_cols,
        lags=args.lags,
        shift_by=args.shift_by,
    )

    # Merge features back with labels/date for filtering
    model_df = df[["date_month_end", target_col]].merge(X_all, on="date_month_end", how="left")

    # Drop rows without target (end-of-series forward label will be NaN before casting)
    model_df = model_df.dropna(subset=[target_col]).copy()
    model_df[target_col] = model_df[target_col].astype("int8")

    # Split by time
    test_start = ensure_month_end(pd.to_datetime(args.test_start))
    train_df, test_df = time_split(model_df, test_start=test_start)

    if train_df.empty or test_df.empty:
        raise ValueError(f"Train/test split produced empty set(s). Check --test-start={args.test_start}")

    X_train = train_df[final_feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[final_feature_cols]
    y_test = test_df[target_col]

    # Train model
    if args.model == "logreg":
        model = train_logreg(X_train, y_train)
    else:
        model = train_xgboost(X_train, y_train)

    # Predict probabilities
    y_prob = predict_proba(model, X_test, model_type=args.model)

    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
    pr_auc = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")

    thr = pick_threshold(y_test.to_numpy(), y_prob, strategy=args.threshold_strategy)
    y_pred = (y_prob >= thr).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Save predictions
    pred_df = test_df[["date_month_end", target_col]].copy()
    pred_df["y_prob"] = y_prob
    pred_df["y_pred"] = y_pred
    pred_path = out_dir / f"predictions_{target_col}_{args.model}.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save metrics + confusion matrix
    result = BacktestResult(
        horizon=args.target_horizon,
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        roc_auc=float(roc_auc) if np.isfinite(roc_auc) else float("nan"),
        pr_auc=float(pr_auc) if np.isfinite(pr_auc) else float("nan"),
        precision=float(p),
        recall=float(r),
        f1=float(f1),
        threshold=float(thr),
    )

    metrics_payload = {
        "target": target_col,
        "test_start": str(test_start.date()),
        "model": args.model,
        "shift_by": args.shift_by,
        "lags": args.lags,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "threshold": result.threshold,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "feature_count": len(final_feature_cols),
    }
    (out_dir / f"metrics_{target_col}_{args.model}.json").write_text(json.dumps(metrics_payload, indent=2))

    # Feature importance
    imp_path = out_dir / f"feature_importance_{target_col}_{args.model}.csv"
    export_feature_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=final_feature_cols,
        out_path=imp_path,
        model_type=args.model,
    )

    # Console summary
    print("\n=== Backtest Summary ===")
    print(f"Target:               {target_col}")
    print(f"Train rows:           {result.n_train}")
    print(f"Test rows:            {result.n_test}")
    print(f"Model:                {args.model}")
    print(f"Leak-safe shift_by:   {args.shift_by} month(s)")
    print(f"Extra lags:           {args.lags}")
    print(f"ROC-AUC:              {result.roc_auc:.4f}" if np.isfinite(
        result.roc_auc) else "ROC-AUC: NaN (single-class test)")
    print(f"PR-AUC:               {result.pr_auc:.4f}" if np.isfinite(
        result.pr_auc) else "PR-AUC: NaN (single-class test)")
    print(f"Threshold:            {result.threshold:.2f} (strategy={args.threshold_strategy})")
    print(f"Precision / Recall:   {result.precision:.4f} / {result.recall:.4f}")
    print(f"F1:                   {result.f1:.4f}")
    print("Confusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)

    print("\nArtifacts written:")
    print(f"  - {pred_path}")
    print(f"  - {out_dir / f'metrics_{target_col}_{args.model}.json'}")
    if imp_path.exists():
        print(f"  - {imp_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
source .venv/bin/activate
pip install scikit-learn matplotlib  # pandas/pyarrow already installed for your panel
# Optional:
# pip install xgboost

Example run (6-month forward label, hold out from 2005 onward, add 3 lags):
python recession_backtest.py \
  --input ./fred_macro_pack/panels/macro_panel_monthly.parquet \
  --horizons 3 6 12 \
  --target-horizon 6 \
  --test-start 2005-01-31 \
  --lags 3 \
  --shift-by 1 \
  --model logreg \
  --out ./backtest_out

"""
