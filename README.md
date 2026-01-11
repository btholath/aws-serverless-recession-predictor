### **1. GitHub Repository Details**

* **Repository Name:** `aws-serverless-recession-predictor`
* **Description:** An end-to-end MLOps pipeline on AWS predicting US economic indicators using FRED data. Features
  SageMaker Model Registry governance and cost-efficient Serverless Inference.

---

### **2. The `.gitignore` File**

Create a file named `.gitignore` in your root directory. This ensures you don't accidentally commit your virtual
environment, local data, or AWS account secrets.

```gitignore
# Python / Environment
__pycache__/
*.pyc
.venv/
venv/
.env

# IDEs
.idea/
.vscode/

# Project Data & Artifacts (Don't commit large datasets)
fred_macro_pack/
*.tar
*.tar.gz
*.csv

# AWS Configuration (Don't commit your specific Account IDs)
src/config.py

# Mac OS
.DS_Store

```

---

### **3. The README.md**

Create a file named `README.md` in your root directory. This contains the professional documentation for anyone viewing
your code.

```markdown
# AWS Serverless Recession Predictor 📉

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AWS](https://img.shields.io/badge/AWS-SageMaker-orange)
![Status](https://img.shields.io/badge/Status-POC-green)

A production-ready **MLOps Proof of Concept (POC)** that predicts US economic trends (Unemployment Rate) using
historical data from the Federal Reserve (FRED).

This project demonstrates a cost-optimized AWS architecture using **SageMaker Serverless Inference**, allowing the
application to scale to zero (costing $0) when not in use.

## 🏗 Architecture

1. **Infrastructure as Code:** CloudFormation creates S3 buckets, IAM Roles, and Model Registry groups.
2. **ETL Layer:** Python scripts ingest FRED macroeconomic data into an S3 Data Lake.
3. **Training Pipeline:** AWS SageMaker spins up ephemeral instances to train an **XGBoost** regressor.
4. **Governance:** Models are registered in the **SageMaker Model Registry** and require manual human approval.
5. **Deployment:** Approved models are deployed to a **Serverless Endpoint** for real-time inference.

## 📂 Project Structure

```text
aws-serverless-recession-predictor/
├── infrastructure/
│   └── template.yaml          # CloudFormation template for AWS resources
├── src/
│   ├── config.py              # (GitIgnored) Project configuration & ARNs
│   ├── data/
│   │   └── upload_data.py     # Uploads local FRED data to S3
│   ├── model/
│   │   ├── train_entrypoint.py # The training & inference logic (runs on AWS)
│   │   ├── trigger_training.py # Orchestrates the training job
│   │   └── deploy_and_test.py  # Deploys Serverless Endpoint & runs predictions
└── requirements.txt

```

## 🚀 Getting Started

### Prerequisites

* AWS CLI configured (`aws configure`) with Administrator permissions.
* Python 3.8+ installed.
* Local data folder `fred_macro_pack/` containing FRED CSV files.

### 1. Installation

Clone the repo and set up your virtual environment:

```bash
git clone [https://github.com/your-username/aws-serverless-recession-predictor.git](https://github.com/your-username/aws-serverless-recession-predictor.git)
cd aws-serverless-recession-predictor

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Deploy Infrastructure

Provision the AWS "Skeleton" (S3, Roles, Registry):

```bash
aws cloudformation deploy \
  --template-file infrastructure/template.yaml \
  --stack-name fred-fintech-stack \
  --parameter-overrides UserEmail="your-email@example.com" \
  --capabilities CAPABILITY_NAMED_IAM

```

**Configuration Step:**

1. Copy `src/config_template.py` to `src/config.py` (if template exists) or create `src/config.py`.
2. Paste the **Bucket Name** and **Role ARN** from the CloudFormation outputs into `src/config.py`.

### 3. Run the Pipeline

**Step A: Ingest Data**
Uploads local CSVs to the S3 bucket.

```bash
python3 -m src.data.upload_data

```

**Step B: Train & Register**
Starts a SageMaker training job. Once finished, the model appears in the Registry as `PendingManualApproval`.

```bash
python3 -m src.model.trigger_training

```

**Step C: Human Approval**

1. Go to **AWS Console > SageMaker > Model Registry**.
2. Open `FredRecessionModels`.
3. Select the latest version -> **Update Status** -> **Approved**.

**Step D: Deploy & Predict**
Deploys the approved model to a Serverless Endpoint and runs a live prediction.

```bash
python3 -m src.model.deploy_and_test

```

*Expected Output:*

> `✅ Endpoint is Ready!`
> `📊 PREDICTION RESULT (Next Month Unemployment Rate): 3.85`

## 🧹 Cleanup

To avoid ongoing storage costs, remove the resources when finished:

1. **Delete Endpoint:**

```bash
aws sagemaker delete-endpoint --endpoint-name fred-serverless-endpoint

```

2. **Delete CloudFormation Stack:**

```bash
aws cloudformation delete-stack --stack-name fred-fintech-stack

```

3. **Empty S3 Bucket:** Manually delete files in the S3 console or via CLI.

## 🛠 Tech Stack

* **Language:** Python 3.10
* **Cloud:** AWS (SageMaker, S3, CloudFormation, IAM)
* **ML Framework:** XGBoost
* **Orchestration:** Boto3 SDK

```

```