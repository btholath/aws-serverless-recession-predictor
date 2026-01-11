"""
Phase 4: Model Training Logic
File: src/model/train.py This is the script that runs inside the SageMaker container. It trains an XGBoost model to predict Unemployment (UNRATE).
"""
import argparse
import os

import joblib
import pandas as pd
import xgboost as xgb


def parse_args():
    parser = argparse.ArgumentParser()
    # SageMaker passes hyperparameters as arguments
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--num_round", type=int, default=50)

    # SageMaker data paths
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("🚀 Starting Training inside SageMaker...")

    # 1. Load Data
    # We assume 'fred_macro_pack.csv' or similar is in the training channel
    # Adjust filename to match what you actually uploaded
    train_path = os.path.join(args.train, "fred_merged_data.csv")

    # Fallback: find first CSV if specific name unknown
    if not os.path.exists(train_path):
        train_files = [f for f in os.listdir(args.train) if f.endswith('.csv')]
        train_path = os.path.join(args.train, train_files[0])

    print(f"Reading data from: {train_path}")
    df = pd.read_csv(train_path)

    # 2. Preprocess (Simple Recession Logic)
    # Target: Next Month's Unemployment Rate
    df['Target'] = df['UNRATE'].shift(-1)
    df = df.dropna()

    # Features: Drop Date and Target
    X = df.drop(columns=['DATE', 'Target'], errors='ignore')
    y = df['Target']

    # 3. Train
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "objective": "reg:squarederror"
    }

    model = xgb.train(params, dtrain, num_boost_round=args.num_round)

    # 4. Save Model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
