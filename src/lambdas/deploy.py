import os

import boto3

sm_client = boto3.client('sagemaker')


def lambda_handler(event, context):
    # EventBridge sends the Model Package ARN in the details
    model_package_arn = event['detail']['ModelPackageArn']
    timestamp = event['time']

    model_name = f"fred-model-{timestamp.replace(':', '-')}"
    ep_config_name = f"fred-serverless-config-{timestamp.replace(':', '-')}"
    endpoint_name = "fred-production-endpoint"  # Fixed name for prod

    # 1. Create Model Object from Registry
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={'ModelPackageName': model_package_arn},
        ExecutionRoleArn=os.environ['ROLE_ARN']
    )

    # 2. Create Serverless Config
    sm_client.create_endpoint_config(
        EndpointConfigName=ep_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'ServerlessConfig': {
                'MemorySizeInMB': 2048,
                'MaxConcurrency': 5
            }
        }]
    )

    # 3. Update or Create Endpoint
    try:
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=ep_config_name
        )
        print(f"Creating new endpoint: {endpoint_name}")
    except sm_client.exceptions.ResourceInUse:
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=ep_config_name
        )
        print(f"Updating existing endpoint: {endpoint_name}")
