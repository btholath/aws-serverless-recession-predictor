"""
Training Entrypoint for SageMaker XGBoost
This script is executed during training on SageMaker.
"""
import argparse
import os
import sys
import shutil

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


# ============================================================
# IMPORTANT: Include inference.py functions here as well
# This ensures model_fn is available for the XGBoost container
# ============================================================
def model_fn(model_dir):
    """Load model for inference - required by SageMaker"""
    print(f"📦 Loading model from: {model_dir}")
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    return model


# ============================================================
# Training Logic
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--min_child_weight", type=int, default=1)
    parser.add_argument("--num_round", type=int, default=50)
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    parser.add_argument("--early_stopping_rounds", type=int, default=10)

    # SageMaker Directory Paths
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_args()


def load_data(train_dir):
    """Load training data from S3 (via SageMaker's channel)"""
    if not train_dir or not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {train_dir}")

    file_path = os.path.join(train_dir, files[0])
    print(f"📂 Reading data from: {file_path}")

    # CSV format: feature1, feature2, ..., target (no headers)
    df = pd.read_csv(file_path, header=None)
    return df


def train_model(args):
    """Main training function"""
    print("🚀 Starting Training Script...")
    print(f"📋 Hyperparameters: max_depth={args.max_depth}, eta={args.eta}, rounds={args.num_round}")

    # Load data
    df = load_data(args.train)
    print(f"📊 Data shape: {df.shape}")

    # Prepare features and target
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]   # Last column is target

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📈 Train: {len(X_train)}, Validation: {len(X_val)}")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # XGBoost parameters
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "objective": args.objective,
        "verbosity": 1,
        "eval_metric": "rmse"
    }

    # Train
    watchlist = [(dtrain, "train"), (dval, "validation")]
    
    print("⏳ Training XGBoost model...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=watchlist,
        early_stopping_rounds=args.early_stopping_rounds
    )

    # Save model
    model_path = os.path.join(args.model_dir, "xgboost-model")
    model.save_model(model_path)
    print(f"✅ Model saved to {model_path}")

    # Log metrics
    print(f"📉 Best iteration: {model.best_iteration}")
    print(f"📊 Best score: {model.best_score}")

    return model


if __name__ == "__main__":
    args = parse_args()
    
    try:
        train_model(args)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
