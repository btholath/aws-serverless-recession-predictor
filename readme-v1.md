# FRED FinTech ML Pipeline

A production-ready MLOps pipeline for predicting unemployment rates using FRED (Federal Reserve Economic Data) indicators on AWS SageMaker.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  FRED API   │────▶│  S3 Bucket  │────▶│ SageMaker Train │
└─────────────┘     └─────────────┘     └────────┬────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Endpoint   │◀────│   Deploy    │◀────│ Model Registry  │
│ (Serverless)│     │   Lambda    │     │ (Approved)      │
└─────────────┘     └─────────────┘     └─────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
# Clone/download the project
cd fred_fintech_app

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure
```

### 2. Setup AWS Infrastructure

```bash
python scripts/setup_aws.py --email your@email.com
```

This creates:
- S3 bucket for data and models
- IAM role for SageMaker
- Model Registry group
- SNS topic for notifications

### 3. Upload Training Data

```bash
python -m src.data.upload_data
```

### 4. Train Model

```bash
python -m src.model.trigger_training
```

This will:
- Start a SageMaker training job
- Train XGBoost model on FRED data
- Register model in Model Registry (PendingManualApproval)

### 5. Approve Model

1. Go to **AWS Console** → **SageMaker** → **Model Registry**
2. Click on **FredRecessionModels**
3. Select the latest version
4. Click **Update Status** → **Approved**

### 6. Deploy Model

```bash
python -m src.model.deploy
```

This creates a **serverless endpoint** (pay only when used).

### 7. Test Predictions

```python
import boto3

runtime = boto3.client('sagemaker-runtime')
response = runtime.invoke_endpoint(
    EndpointName='fred-serverless-endpoint',
    ContentType='text/csv',
    Body='250.0, 5.25, -0.5, 102.5'  # CPI, FEDFUNDS, T10Y2Y, INDPRO
)
print(response['Body'].read().decode())
```

## Cleanup

To delete all AWS resources:

```bash
python scripts/cleanup_aws.py
```

## Project Structure

```
fred_fintech_app/
├── scripts/
│   ├── setup_aws.py       # Create AWS resources
│   └── cleanup_aws.py     # Delete AWS resources
├── src/
│   ├── config.py          # Configuration
│   ├── data/
│   │   └── upload_data.py # Upload training data to S3
│   └── model/
│       ├── train_entrypoint.py  # Training script (runs on SageMaker)
│       ├── inference.py         # Inference script (runs on endpoint)
│       ├── trigger_training.py  # Start training job
│       └── deploy.py            # Deploy to endpoint
└── requirements.txt
```

## Key Files Explained

| File | Purpose |
|------|---------|
| `train_entrypoint.py` | Runs during training on SageMaker. Trains XGBoost model. |
| `inference.py` | Runs on the endpoint. Handles model loading and predictions. |
| `trigger_training.py` | Local script to start a training job. |
| `deploy.py` | Local script to deploy approved model. |

## Costs

- **Training**: ~$0.10 per training job (ml.m5.large, ~3 min)
- **Endpoint**: $0 when idle (serverless), ~$0.001 per request
- **S3**: Minimal (< $0.01/month for this data)

## Troubleshooting

### Model fails to load on endpoint

**Error**: `Please provide a model_fn implementation`

**Solution**: The `inference.py` file must be included in the model artifact. Retrain the model after adding `inference.py` to `src/model/`.

### Endpoint stuck in "Creating" status

Check CloudWatch Logs:
1. Go to **CloudWatch** → **Log Groups**
2. Find `/aws/sagemaker/Endpoints/fred-serverless-endpoint`
3. Check for errors

### Training job fails

Check the training job logs:
```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

## License

MIT
