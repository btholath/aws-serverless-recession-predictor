"""
Trigger SageMaker Training Job
This script starts the training job and registers the model in Model Registry.
"""
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost

from src.config import BUCKET_NAME, ROLE_ARN, MODEL_PACKAGE_GROUP, REGION


def trigger_training():
    """Start SageMaker training job and register model"""
    
    session = sagemaker.Session()
    
    print("🚀 Starting Training Job...")
    print(f"   Bucket: {BUCKET_NAME}")
    print(f"   Role: {ROLE_ARN}")
    print(f"   Model Group: {MODEL_PACKAGE_GROUP}")

    # 1. Configure the XGBoost Estimator
    xgb_estimator = XGBoost(
        entry_point="train_entrypoint.py",
        source_dir="src/model/",  # This includes train_entrypoint.py AND inference.py
        framework_version="1.5-1",
        py_version="py3",
        hyperparameters={
            "max_depth": "5",
            "eta": "0.2",
            "num_round": "30",
            "objective": "reg:squarederror",
            "early_stopping_rounds": "10"
        },
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=f"s3://{BUCKET_NAME}/models/output",
        sagemaker_session=session
    )

    # 2. Define Training Input
    train_input = TrainingInput(
        s3_data=f"s3://{BUCKET_NAME}/fred-data/train/train.csv",
        content_type="csv"
    )

    # 3. Start Training (waits for completion)
    print("⏳ Training in progress...")
    xgb_estimator.fit({"train": train_input}, wait=True)
    print("✅ Training completed!")

    # 4. Register Model in Model Registry
    print("📝 Registering model in Model Registry...")
    model_package = xgb_estimator.register(
        model_package_group_name=MODEL_PACKAGE_GROUP,
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        content_types=["text/csv", "application/json"],
        response_types=["text/csv", "application/json"],
        approval_status="PendingManualApproval",
        description="FRED Unemployment Prediction Model"
    )

    print(f"✅ Model Registered: {model_package.model_package_arn}")
    print("")
    print("📋 Next Steps:")
    print("   1. Go to SageMaker Console → Model Registry → FredRecessionModels")
    print("   2. Click on the new version and change status to 'Approved'")
    print("   3. Run: python -m src.model.deploy")
    
    return model_package.model_package_arn


if __name__ == "__main__":
    trigger_training()
