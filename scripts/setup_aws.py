#!/usr/bin/env python3
"""
AWS Infrastructure Setup Script
Creates all required AWS resources for the FRED FinTech ML Pipeline.

Usage:
    python scripts/setup_aws.py --email your@email.com
"""
import argparse
import time
import boto3
from botocore.exceptions import ClientError


def get_account_id():
    return boto3.client('sts').get_caller_identity()['Account']


def create_s3_bucket(bucket_name, region):
    """Create S3 bucket for data and model artifacts"""
    print(f"📦 Creating S3 bucket: {bucket_name}")
    s3 = boto3.client('s3', region_name=region)
    
    try:
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"   ✅ Created: {bucket_name}")
        return True
    except ClientError as e:
        if 'BucketAlreadyOwnedByYou' in str(e):
            print(f"   ℹ️ Bucket already exists (owned by you)")
            return True
        elif 'BucketAlreadyExists' in str(e):
            print(f"   ❌ Bucket name taken by another account")
            return False
        raise


def create_sagemaker_role(role_name, account_id):
    """Create IAM role for SageMaker"""
    print(f"🔐 Creating IAM Role: {role_name}")
    iam = boto3.client('iam')
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=str(trust_policy).replace("'", '"'),
            Description="SageMaker execution role for FRED ML Pipeline"
        )
        role_arn = response['Role']['Arn']
        
        # Attach required policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
        ]
        for policy in policies:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy)
        
        print(f"   ✅ Created: {role_arn}")
        # Wait for role to propagate
        time.sleep(10)
        return role_arn
        
    except ClientError as e:
        if 'EntityAlreadyExists' in str(e):
            role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
            print(f"   ℹ️ Role already exists: {role_arn}")
            return role_arn
        raise


def create_model_package_group(group_name, region):
    """Create SageMaker Model Package Group"""
    print(f"📋 Creating Model Package Group: {group_name}")
    sm = boto3.client('sagemaker', region_name=region)
    
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription="FRED Economic Indicator Models"
        )
        print(f"   ✅ Created: {group_name}")
        return True
    except ClientError as e:
        if 'ResourceInUse' in str(e) or 'already exists' in str(e).lower():
            print(f"   ℹ️ Model Package Group already exists")
            return True
        raise


def create_sns_topic(topic_name, email, region):
    """Create SNS topic for model approval notifications"""
    print(f"📧 Creating SNS Topic: {topic_name}")
    sns = boto3.client('sns', region_name=region)
    
    try:
        response = sns.create_topic(Name=topic_name)
        topic_arn = response['TopicArn']
        
        # Subscribe email
        if email:
            sns.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email
            )
            print(f"   ✅ Created: {topic_arn}")
            print(f"   📬 Subscription sent to: {email}")
            print(f"   ⚠️  Check your email and confirm the subscription!")
        
        return topic_arn
    except ClientError as e:
        print(f"   ❌ Failed: {e}")
        return None


def setup_infrastructure(email, region='us-east-1'):
    """Main setup function"""
    print("=" * 60)
    print("🚀 FRED FinTech ML Pipeline - AWS Setup")
    print("=" * 60)
    
    account_id = get_account_id()
    print(f"📋 Account ID: {account_id}")
    print(f"🌎 Region: {region}")
    print()
    
    # Configuration
    bucket_name = f"fred-fintech-data-{account_id}"
    role_name = "fred-sagemaker-role"
    model_group = "FredRecessionModels"
    topic_name = "FredModelApproval"
    
    # Create resources
    create_s3_bucket(bucket_name, region)
    role_arn = create_sagemaker_role(role_name, account_id)
    create_model_package_group(model_group, region)
    topic_arn = create_sns_topic(topic_name, email, region)
    
    # Print summary
    print()
    print("=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print()
    print("📋 Resource Summary:")
    print(f"   S3 Bucket:    {bucket_name}")
    print(f"   IAM Role:     {role_arn}")
    print(f"   Model Group:  {model_group}")
    print(f"   SNS Topic:    {topic_arn}")
    print()
    print("📝 Update src/config.py with these values:")
    print(f'   BUCKET_NAME = "{bucket_name}"')
    print(f'   ROLE_ARN = "{role_arn}"')
    print()
    print("🚀 Next Steps:")
    print("   1. Confirm SNS email subscription")
    print("   2. Upload training data: python -m src.data.upload_data")
    print("   3. Start training: python -m src.model.trigger_training")


def main():
    parser = argparse.ArgumentParser(description='Setup AWS infrastructure')
    parser.add_argument('--email', required=True, help='Email for notifications')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    args = parser.parse_args()
    
    setup_infrastructure(args.email, args.region)


if __name__ == "__main__":
    main()
