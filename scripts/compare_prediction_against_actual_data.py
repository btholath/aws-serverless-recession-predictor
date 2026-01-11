import boto3
import pandas as pd

# Load your test data
df = pd.read_csv('fred_dataset/fred_macro_pack/panels/macro_panel_monthly.csv')

# Get last 10 rows for testing
test_data = df[['CPIAUCSL', 'FEDFUNDS', 'T10Y2Y', 'INDPRO', 'UNRATE']].tail(10).dropna()

runtime = boto3.client('sagemaker-runtime')

print("Comparing Predicted vs Actual:\n")
print(f"{'Actual':>10} {'Predicted':>10} {'Error':>10}")
print("-" * 35)

for _, row in test_data.iterrows():
    payload = f"{row['CPIAUCSL']}, {row['FEDFUNDS']}, {row['T10Y2Y']}, {row['INDPRO']}"

    response = runtime.invoke_endpoint(
        EndpointName='fred-serverless-endpoint',
        ContentType='text/csv',
        Body=payload
    )

    predicted = float(response['Body'].read().decode().strip('[]'))
    actual = row['UNRATE']
    error = abs(predicted - actual)

    print(f"{actual:>10.2f} {predicted:>10.2f} {error:>10.2f}")
