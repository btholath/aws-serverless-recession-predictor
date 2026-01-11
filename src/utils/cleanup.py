import boto3
from botocore.exceptions import ClientError
from src.config import BUCKET_NAME, MODEL_PACKAGE_GROUP, REGION

# Initialize Clients
s3 = boto3.resource('s3', region_name=REGION)
sm_client = boto3.client('sagemaker', region_name=REGION)
cf_client = boto3.client('cloudformation', region_name=REGION)

STACK_NAME = "fred-fintech-stack"  # Must match what you used in Phase 2


def delete_endpoints():
    print("🔻 Finding SageMaker Endpoints...")
    # List endpoints containing 'fred' or 'serverless'
    endpoints = sm_client.list_endpoints(NameContains='fred')

    for ep in endpoints['Endpoints']:
        ep_name = ep['EndpointName']
        print(f"   Deleting Endpoint: {ep_name}")
        sm_client.delete_endpoint(EndpointName=ep_name)

        # Also delete the Config
        config_name = ep['EndpointConfigName']
        print(f"   Deleting Config: {config_name}")
        sm_client.delete_endpoint_config(EndpointConfigName=config_name)


def empty_and_delete_bucket():
    print(f"🔻 Emptying S3 Bucket: {BUCKET_NAME}...")
    try:
        bucket = s3.Bucket(BUCKET_NAME)
        # Delete all versions and objects
        bucket.object_versions.delete()
        bucket.objects.all().delete()

        print(f"   Deleting Bucket...")
        bucket.delete()
        print("   ✅ Bucket Deleted")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            print("   Bucket already gone.")
        else:
            print(f"   Error: {e}")


def delete_model_registry():
    print(f"🔻 Cleaning Model Registry: {MODEL_PACKAGE_GROUP}...")
    try:
        # 1. List all model versions
        packages = sm_client.list_model_packages(ModelPackageGroupName=MODEL_PACKAGE_GROUP)

        for pkg in packages['ModelPackageSummaryList']:
            arn = pkg['ModelPackageArn']
            print(f"   Deleting Model Version: {arn.split('/')[-1]}")
            sm_client.delete_model_package(ModelPackageName=arn)

        # 2. Delete the Group (CloudFormation usually does this, but we clean up to be safe)
        # sm_client.delete_model_package_group(ModelPackageGroupName=MODEL_PACKAGE_GROUP)
        # print("   ✅ Model Group Deleted")

    except ClientError as e:
        print(f"   Registry cleanup skipped or failed: {e}")


def delete_cloudformation():
    print(f"🔻 Deleting CloudFormation Stack: {STACK_NAME}...")
    try:
        cf_client.delete_stack(StackName=STACK_NAME)
        print("   ⏳ Stack deletion initiated. This removes Roles, Topics, and Registry Groups.")
    except Exception as e:
        print(f"   Error deleting stack: {e}")


if __name__ == "__main__":
    print("⚠️  WARNING: STARTING RESOURCE TEARDOWN ⚠️")
    confirm = input("Type 'DELETE' to confirm destruction of all resources: ")

    if confirm == "DELETE":
        delete_endpoints()
        delete_model_registry()  # Clean versions first so CFN can delete the group
        empty_and_delete_bucket()
        delete_cloudformation()
        print("\n✅ Teardown Complete. Check CloudFormation console for final stack status.")
    else:
        print("❌ Aborted.")
