import datetime
from io import StringIO

import boto3
import pandas_datareader.data as web
from src.config import BUCKET_NAME, REGION  # Assuming you set these in config.py


def run_ingestion():
    # 1. Define Macro Series
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime.now()
    series_map = {
        'CPIAUCSL': 'CPI',
        'UNRATE': 'Unemployment',
        'FEDFUNDS': 'FedFunds',
        'T10Y2Y': 'YieldSpread'
    }

    # 2. Fetch Data
    print(f"Fetching FRED data from {start} to {end}...")
    df = web.DataReader(list(series_map.keys()), 'fred', start, end)
    df = df.rename(columns=series_map).ffill().dropna()

    # 3. Target Creation (Predict Next Month's Unemployment)
    df['Target'] = df['Unemployment'].shift(-1)
    df = df.dropna()

    # 4. Split and Upload
    train_data = df.iloc[:-12]

    csv_buffer = StringIO()
    train_data.to_csv(csv_buffer, header=False, index=False)

    s3 = boto3.client('s3', region_name=REGION)
    key = "fred-data/train/train.csv"

    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=csv_buffer.getvalue())
    print(f"✅ Data uploaded to s3://{BUCKET_NAME}/{key}")


if __name__ == "__main__":
    run_ingestion()
