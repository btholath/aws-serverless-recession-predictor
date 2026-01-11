"""
Upload Training Data to S3
Prepares and uploads the training data for SageMaker.
"""
import os
import boto3
import pandas as pd
from src.config import BUCKET_NAME, S3_DATA_PREFIX


def prepare_training_data(input_file, output_file):
    """
    Prepare data for XGBoost training.
    Format: feature1, feature2, ..., target (no headers)
    """
    print(f"📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Select features and target
    # Adjust these columns based on your dataset
    feature_columns = ['CPIAUCSL', 'FEDFUNDS', 'T10Y2Y', 'INDPRO']
    target_column = 'UNRATE'
    
    # Check if columns exist
    available_cols = df.columns.tolist()
    print(f"📋 Available columns: {available_cols}")
    
    # Use available columns
    features = [c for c in feature_columns if c in available_cols]
    if target_column not in available_cols:
        print(f"⚠️  Target column '{target_column}' not found")
        # Try to find an alternative
        for alt in ['UNRATE', 'unemployment', 'target']:
            if alt in available_cols:
                target_column = alt
                break
    
    if not features:
        raise ValueError(f"No feature columns found. Available: {available_cols}")
    
    print(f"📊 Using features: {features}")
    print(f"🎯 Target: {target_column}")
    
    # Create training dataframe
    df_train = df[features + [target_column]].dropna()
    
    # Save without headers (SageMaker XGBoost format)
    df_train.to_csv(output_file, index=False, header=False)
    print(f"✅ Saved {len(df_train)} rows to: {output_file}")
    
    return output_file


def upload_to_s3(local_file, bucket, s3_key):
    """Upload file to S3"""
    print(f"☁️ Uploading to s3://{bucket}/{s3_key}")
    s3 = boto3.client('s3')
    s3.upload_file(local_file, bucket, s3_key)
    print(f"✅ Upload complete!")
    return f"s3://{bucket}/{s3_key}"


def main():
    """Main upload flow"""
    print("=" * 60)
    print("📤 FRED Data Upload")
    print("=" * 60)
    
    # Find input data file
    possible_paths = [
        'fred_dataset/fred_macro_pack/panels/macro_panel_monthly.csv',
        'data/train.csv',
        'train.csv'
    ]
    
    input_file = None
    for path in possible_paths:
        if os.path.exists(path):
            input_file = path
            break
    
    if not input_file:
        print("❌ No training data found!")
        print("   Please ensure you have a data file at one of:")
        for p in possible_paths:
            print(f"   - {p}")
        return
    
    # Prepare data
    output_file = '/tmp/train.csv'
    prepare_training_data(input_file, output_file)
    
    # Upload to S3
    s3_key = f"{S3_DATA_PREFIX}/train.csv"
    s3_uri = upload_to_s3(output_file, BUCKET_NAME, s3_key)
    
    print()
    print("=" * 60)
    print("✅ Data Upload Complete!")
    print("=" * 60)
    print(f"   S3 URI: {s3_uri}")
    print()
    print("🚀 Next: python -m src.model.trigger_training")


if __name__ == "__main__":
    main()
