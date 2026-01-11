"""
Step 2: The "Engineer" Role (Deploy & Predict)
Now that the model is approved, we can deploy it. We will use a Serverless Endpoint to keep costs at $0 when you aren't using it.
"""
import time

import boto3
from src.config import ROLE_ARN, MODEL_PACKAGE_GROUP, REGION

sm_client = boto3.client('sagemaker', region_name=REGION)
runtime_client = boto3.client('sagemaker-runtime', region_name=REGION)

ENDPOINT_NAME = "fred-serverless-endpoint"


def get_latest_approved_model():
    """Finds the latest Approved model version ARN."""
    print(f"🔍 Searching for latest approved model in {MODEL_PACKAGE_GROUP}...")

    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        SortBy="CreationTime",
        SortOrder="Descending"
    )

    for pkg in response['ModelPackageSummaryList']:
        if pkg['ModelApprovalStatus'] == "Approved":
            print(f"✅ Found Approved Model: {pkg['ModelPackageArn']}")
            return pkg['ModelPackageArn']

    raise Exception("❌ No Approved Model found! Did you go to the Console and Approve it?")


def deploy_serverless_endpoint(model_package_arn):
    """Deploys the model to a Serverless Endpoint."""

    # 1. Create Model Object
    model_name = "fred-model-" + str(int(time.time()))
    print(f"🏗️ Creating Model: {model_name}")
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={'ModelPackageName': model_package_arn},
        ExecutionRoleArn=ROLE_ARN
    )

    # 2. Create Endpoint Config (Serverless)
    config_name = "fred-serverless-config-" + str(int(time.time()))
    print(f"⚙️ Creating Configuration: {config_name}")
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'ServerlessConfig': {
                'MemorySizeInMB': 2048,
                'MaxConcurrency': 5
            }
        }]
    )

    # 3. Create Endpoint
    print(f"🚀 Deploying Endpoint: {ENDPOINT_NAME} (This takes ~2-3 mins)...")
    try:
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name
        )
    except sm_client.exceptions.ResourceInUse:
        print(f"   Endpoint {ENDPOINT_NAME} already exists. Updating it...")
        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name
        )

    # 4. Wait for it to be ready
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=ENDPOINT_NAME)
    print("✅ Endpoint is Ready!")


def test_prediction():
    """Sends a sample payload to the endpoint."""
    print("🔮 Testing Prediction...")

    # Sample Data: [CPI, FEDFUNDS, T10Y2Y] (Use real values from your CSV logic)
    # This must match the order of columns your model was trained on!
    sample_payload = "250.0, 5.25, -0.5, 102.5"  # Example CSV string

    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='text/csv',
        Body=sample_payload
    )

    result = response['Body'].read().decode('utf-8')
    print(f"\n📊 PREDICTION RESULT (Next Month Unemployment Rate): {result}")


if __name__ == "__main__":
    arn = get_latest_approved_model()
    deploy_serverless_endpoint(arn)
    test_prediction()
