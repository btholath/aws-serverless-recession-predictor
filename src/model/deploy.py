"""
Deploy Model to SageMaker Serverless Endpoint
This script deploys an approved model from Model Registry.
"""
import time
import boto3
import tarfile
import os
import tempfile

from src.config import (
    ROLE_ARN, MODEL_PACKAGE_GROUP, REGION, 
    ENDPOINT_NAME, BUCKET_NAME
)

sm_client = boto3.client('sagemaker', region_name=REGION)
s3_client = boto3.client('s3', region_name=REGION)
runtime_client = boto3.client('sagemaker-runtime', region_name=REGION)


def get_latest_approved_model():
    """Find the latest Approved model version ARN"""
    print(f"🔍 Searching for approved model in {MODEL_PACKAGE_GROUP}...")

    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        SortBy="CreationTime",
        SortOrder="Descending"
    )

    for pkg in response.get('ModelPackageSummaryList', []):
        if pkg['ModelApprovalStatus'] == "Approved":
            print(f"✅ Found: {pkg['ModelPackageArn']}")
            return pkg['ModelPackageArn']

    raise Exception(
        "❌ No Approved Model found!\n"
        "   Go to SageMaker Console → Model Registry → FredRecessionModels\n"
        "   Select the latest version and change status to 'Approved'"
    )


def get_model_artifact_uri(model_package_arn):
    """Get the S3 URI of the model artifact from the model package"""
    response = sm_client.describe_model_package(ModelPackageName=model_package_arn)
    
    # Get the model data URL from inference specification
    containers = response['InferenceSpecification']['Containers']
    model_data_url = containers[0]['ModelDataUrl']
    
    print(f"📦 Model artifact: {model_data_url}")
    return model_data_url


def create_inference_tar():
    """Create a tarball with inference code"""
    print("📝 Creating inference code package...")
    
    # Get the path to inference.py
    inference_script = os.path.join(
        os.path.dirname(__file__), 
        'inference.py'
    )
    
    if not os.path.exists(inference_script):
        # Create a minimal inference.py
        inference_code = '''
import os
import xgboost as xgb
import numpy as np

def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    return model

def input_fn(request_body, content_type):
    if content_type == "text/csv":
        values = [float(x.strip()) for x in request_body.strip().split(",")]
        return xgb.DMatrix(np.array([values]))
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, accept):
    return str(prediction[0]) if len(prediction) == 1 else ",".join(map(str, prediction))
'''
        # Write to temp file
        inference_script = os.path.join(tempfile.gettempdir(), 'inference.py')
        with open(inference_script, 'w') as f:
            f.write(inference_code)
    
    # Create tar.gz
    tar_path = os.path.join(tempfile.gettempdir(), 'sourcedir.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(inference_script, arcname='inference.py')
    
    # Upload to S3
    s3_key = 'inference-code/sourcedir.tar.gz'
    s3_client.upload_file(tar_path, BUCKET_NAME, s3_key)
    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"
    
    print(f"✅ Inference code uploaded: {s3_uri}")
    return s3_uri


def deploy_serverless_endpoint(model_package_arn):
    """Deploy model to a Serverless Endpoint"""
    
    timestamp = str(int(time.time()))
    model_name = f"fred-model-{timestamp}"
    config_name = f"fred-config-{timestamp}"

    # Get model package details
    pkg_response = sm_client.describe_model_package(ModelPackageName=model_package_arn)
    container = pkg_response['InferenceSpecification']['Containers'][0]
    
    # 1. Create Model
    print(f"🏗️ Creating Model: {model_name}")
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'ModelPackageName': model_package_arn
        },
        ExecutionRoleArn=ROLE_ARN
    )

    # 2. Create Endpoint Config (Serverless)
    print(f"⚙️ Creating Config: {config_name}")
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

    # 3. Create or Update Endpoint
    print(f"🚀 Deploying Endpoint: {ENDPOINT_NAME}")
    try:
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name
        )
    except sm_client.exceptions.ResourceInUse:
        print("   Endpoint exists, updating...")
        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name
        )

    # 4. Wait for deployment
    print("⏳ Waiting for endpoint to be ready (2-5 mins)...")
    waiter = sm_client.get_waiter('endpoint_in_service')
    try:
        waiter.wait(
            EndpointName=ENDPOINT_NAME,
            WaiterConfig={'Delay': 30, 'MaxAttempts': 20}
        )
        print("✅ Endpoint is ready!")
        return True
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        # Get failure reason
        response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"   Status: {response['EndpointStatus']}")
        if 'FailureReason' in response:
            print(f"   Reason: {response['FailureReason']}")
        return False


def test_endpoint():
    """Test the deployed endpoint with sample data"""
    print("\n🔮 Testing Endpoint...")
    
    # Sample payload: CPI, FEDFUNDS, T10Y2Y, INDPRO (adjust to your features)
    sample_payload = "250.0, 5.25, -0.5, 102.5"
    
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=sample_payload
        )
        
        result = response['Body'].read().decode('utf-8')
        print(f"📊 Input: {sample_payload}")
        print(f"📈 Prediction (Unemployment Rate): {result}")
        return result
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None


def main():
    """Main deployment flow"""
    print("=" * 60)
    print("🚀 FRED Model Deployment")
    print("=" * 60)
    
    # Step 1: Get approved model
    model_arn = get_latest_approved_model()
    
    # Step 2: Deploy
    success = deploy_serverless_endpoint(model_arn)
    
    if success:
        # Step 3: Test
        test_endpoint()
        
        print("\n" + "=" * 60)
        print("✅ Deployment Complete!")
        print(f"   Endpoint: {ENDPOINT_NAME}")
        print("=" * 60)
    else:
        print("\n❌ Deployment failed. Check CloudWatch logs for details.")


if __name__ == "__main__":
    main()
