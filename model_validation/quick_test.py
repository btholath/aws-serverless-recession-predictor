import boto3

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Test prediction
response = runtime.invoke_endpoint(
    EndpointName='fred-serverless-endpoint',
    ContentType='text/csv',
    Body='250.0, 5.25, -0.5, 102.5'  # CPI, FEDFUNDS, T10Y2Y, INDPRO
)

result = response['Body'].read().decode()
print(f"Predicted Unemployment Rate: {result}")
