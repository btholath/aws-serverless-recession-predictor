"""
Phase 5: The Pipeline Orchestrator
File: src/model/pipeline.py This script submits the job to AWS and registers the result.
"""
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost
from src.config import BUCKET_NAME, ROLE_ARN, S3_DATA_PREFIX, S3_MODEL_OUTPUT, MODEL_PACKAGE_GROUP


def run_pipeline():
    print("🏗️ Building SageMaker Estimator...")

    # 1. Define Estimator
    xgb = XGBoost(
        entry_point="train.py",
        source_dir="src/model",  # Zips this folder and sends to AWS
        framework_version="1.5-1",
        py_version="py3",
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.large",  # Cost-effective training
        output_path=S3_MODEL_OUTPUT,
        hyperparameters={"max_depth": 5, "num_round": 50}
    )

    # 2. Define Data Input
    s3_input = TrainingInput(f"s3://{BUCKET_NAME}/{S3_DATA_PREFIX}", content_type="csv")

    # 3. Train
    print("⏳ Starting Training Job (this takes ~3-5 mins)...")
    xgb.fit({'train': s3_input}, wait=True)

    # 4. Register
    print("📝 Registering Model...")
    model_package = xgb.register(
        model_package_group_name=MODEL_PACKAGE_GROUP,
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        content_types=["text/csv"],
        response_types=["text/csv"],
        approval_status="PendingManualApproval",  # <--- Triggers the EventBridge
        description="FRED Recession Predictor v1"
    )

    print(f"🎉 Model Registered! ARN: {model_package.model_package_arn}")


if __name__ == "__main__":
    run_pipeline()
