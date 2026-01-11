#!/usr/bin/env python3
"""
AWS Infrastructure Cleanup Script
Removes all AWS resources created for the FRED FinTech ML Pipeline.

Usage:
    python scripts/cleanup_aws.py
    python scripts/cleanup_aws.py --force  # Skip confirmation prompts
"""
import argparse
import boto3
from botocore.exceptions import ClientError


def get_account_id():
    return boto3.client('sts').get_caller_identity()['Account']


def delete_endpoint(endpoint_name, region):
    """Delete SageMaker endpoint and its configuration"""
    print(f"🗑️ Deleting Endpoint: {endpoint_name}")
    sm = boto3.client('sagemaker', region_name=region)
    
    try:
        # Get endpoint config name
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        config_name = response['EndpointConfigName']
        
        # Delete endpoint
        sm.delete_endpoint(EndpointName=endpoint_name)
        print(f"   ✅ Deleted endpoint: {endpoint_name}")
        
        # Wait for deletion
        print("   ⏳ Waiting for endpoint deletion...")
        waiter = sm.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name)
        
        # Delete endpoint config
        try:
            sm.delete_endpoint_config(EndpointConfigName=config_name)
            print(f"   ✅ Deleted config: {config_name}")
        except:
            pass
            
    except ClientError as e:
        if 'Could not find' in str(e) or 'does not exist' in str(e).lower():
            print(f"   ℹ️ Endpoint not found (already deleted)")
        else:
            print(f"   ⚠️ Warning: {e}")


def delete_models(prefix, region):
    """Delete SageMaker models matching prefix"""
    print(f"🗑️ Deleting Models with prefix: {prefix}")
    sm = boto3.client('sagemaker', region_name=region)
    
    try:
        response = sm.list_models(NameContains=prefix, MaxResults=100)
        models = response.get('Models', [])
        
        if not models:
            print(f"   ℹ️ No models found")
            return
            
        for model in models:
            model_name = model['ModelName']
            try:
                sm.delete_model(ModelName=model_name)
                print(f"   ✅ Deleted: {model_name}")
            except Exception as e:
                print(f"   ⚠️ Could not delete {model_name}: {e}")
                
    except Exception as e:
        print(f"   ⚠️ Warning: {e}")


def delete_model_packages(group_name, region):
    """Delete all model packages in a group"""
    print(f"🗑️ Deleting Model Packages in: {group_name}")
    sm = boto3.client('sagemaker', region_name=region)
    
    try:
        response = sm.list_model_packages(
            ModelPackageGroupName=group_name,
            MaxResults=100
        )
        packages = response.get('ModelPackageSummaryList', [])
        
        for pkg in packages:
            arn = pkg['ModelPackageArn']
            try:
                sm.delete_model_package(ModelPackageName=arn)
                print(f"   ✅ Deleted: {arn.split('/')[-1]}")
            except Exception as e:
                print(f"   ⚠️ Could not delete: {e}")
                
    except Exception as e:
        print(f"   ⚠️ Warning: {e}")


def delete_model_package_group(group_name, region):
    """Delete Model Package Group"""
    print(f"🗑️ Deleting Model Package Group: {group_name}")
    sm = boto3.client('sagemaker', region_name=region)
    
    # First delete all packages in the group
    delete_model_packages(group_name, region)
    
    try:
        sm.delete_model_package_group(ModelPackageGroupName=group_name)
        print(f"   ✅ Deleted: {group_name}")
    except ClientError as e:
        if 'does not exist' in str(e).lower():
            print(f"   ℹ️ Group not found")
        else:
            print(f"   ⚠️ Warning: {e}")


def delete_s3_bucket(bucket_name, region):
    """Delete S3 bucket and all its contents"""
    print(f"🗑️ Deleting S3 Bucket: {bucket_name}")
    s3 = boto3.resource('s3', region_name=region)
    
    try:
        bucket = s3.Bucket(bucket_name)
        
        # Delete all objects
        print("   ⏳ Deleting objects...")
        bucket.objects.all().delete()
        
        # Delete all versions (for versioned buckets)
        bucket.object_versions.all().delete()
        
        # Delete bucket
        bucket.delete()
        print(f"   ✅ Deleted: {bucket_name}")
        
    except ClientError as e:
        if 'NoSuchBucket' in str(e):
            print(f"   ℹ️ Bucket not found")
        else:
            print(f"   ⚠️ Warning: {e}")


def delete_iam_role(role_name):
    """Delete IAM role and its attached policies"""
    print(f"🗑️ Deleting IAM Role: {role_name}")
    iam = boto3.client('iam')
    
    try:
        # Detach managed policies
        response = iam.list_attached_role_policies(RoleName=role_name)
        for policy in response.get('AttachedPolicies', []):
            iam.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
            print(f"   ✅ Detached: {policy['PolicyName']}")
        
        # Delete inline policies
        response = iam.list_role_policies(RoleName=role_name)
        for policy_name in response.get('PolicyNames', []):
            iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
        
        # Delete role
        iam.delete_role(RoleName=role_name)
        print(f"   ✅ Deleted: {role_name}")
        
    except ClientError as e:
        if 'NoSuchEntity' in str(e):
            print(f"   ℹ️ Role not found")
        else:
            print(f"   ⚠️ Warning: {e}")


def delete_sns_topic(topic_name, region):
    """Delete SNS topic"""
    print(f"🗑️ Deleting SNS Topic: {topic_name}")
    sns = boto3.client('sns', region_name=region)
    account_id = get_account_id()
    
    topic_arn = f"arn:aws:sns:{region}:{account_id}:{topic_name}"
    
    try:
        sns.delete_topic(TopicArn=topic_arn)
        print(f"   ✅ Deleted: {topic_name}")
    except ClientError as e:
        if 'NotFound' in str(e):
            print(f"   ℹ️ Topic not found")
        else:
            print(f"   ⚠️ Warning: {e}")


def delete_endpoint_configs(prefix, region):
    """Delete orphaned endpoint configurations"""
    print(f"🗑️ Deleting Endpoint Configs with prefix: {prefix}")
    sm = boto3.client('sagemaker', region_name=region)
    
    try:
        response = sm.list_endpoint_configs(NameContains=prefix, MaxResults=100)
        configs = response.get('EndpointConfigs', [])
        
        for config in configs:
            config_name = config['EndpointConfigName']
            try:
                sm.delete_endpoint_config(EndpointConfigName=config_name)
                print(f"   ✅ Deleted: {config_name}")
            except:
                pass
    except:
        pass


def cleanup_infrastructure(force=False, region='us-east-1'):
    """Main cleanup function"""
    print("=" * 60)
    print("🧹 FRED FinTech ML Pipeline - AWS Cleanup")
    print("=" * 60)
    
    account_id = get_account_id()
    print(f"📋 Account ID: {account_id}")
    print(f"🌎 Region: {region}")
    print()
    
    # Configuration
    bucket_name = f"fred-fintech-data-{account_id}"
    role_name = "fred-sagemaker-role"
    model_group = "FredRecessionModels"
    endpoint_name = "fred-serverless-endpoint"
    topic_name = "FredModelApproval"
    
    if not force:
        print("⚠️  This will DELETE the following resources:")
        print(f"   - Endpoint: {endpoint_name}")
        print(f"   - Models: fred-model-*")
        print(f"   - Model Group: {model_group}")
        print(f"   - S3 Bucket: {bucket_name} (ALL DATA!)")
        print(f"   - IAM Role: {role_name}")
        print(f"   - SNS Topic: {topic_name}")
        print()
        confirm = input("Type 'DELETE' to confirm: ")
        if confirm != 'DELETE':
            print("❌ Cleanup cancelled")
            return
        print()
    
    # Delete resources in order (dependencies first)
    delete_endpoint(endpoint_name, region)
    delete_endpoint_configs("fred", region)
    delete_models("fred", region)
    delete_model_package_group(model_group, region)
    delete_s3_bucket(bucket_name, region)
    delete_iam_role(role_name)
    delete_sns_topic(topic_name, region)
    
    print()
    print("=" * 60)
    print("✅ Cleanup Complete!")
    print("=" * 60)
    print()
    print("All FRED ML Pipeline resources have been deleted.")


def main():
    parser = argparse.ArgumentParser(description='Cleanup AWS infrastructure')
    parser.add_argument('--force', action='store_true', help='Skip confirmation')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    args = parser.parse_args()
    
    cleanup_infrastructure(args.force, args.region)


if __name__ == "__main__":
    main()
