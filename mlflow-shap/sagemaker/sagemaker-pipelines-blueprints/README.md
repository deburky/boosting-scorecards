# SageMaker Pipelines Blueprints

⚠️ **IMPORTANT: All AWS account IDs, ARNs, and credentials in this directory are EXAMPLE/PLACEHOLDER values only. Replace them with your actual AWS resources before use. See [SECURITY.md](SECURITY.md) for details.**

This directory contains generic, reusable blueprints for building end-to-end ML pipelines on AWS SageMaker. These templates demonstrate best practices for automating data preparation, model training, batch inference, and data warehouse integration.

## Overview

The blueprints implement a complete ML pipeline with four main steps:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data           │     │  Model          │     │  Batch          │     │  Redshift       │
│  Preparation    │────▶│  Training       │────▶│  Inference      │────▶│  Load           │
│                 │     │                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Pipeline Steps

1. **Data Preparation** - ETL, feature engineering, train-test split, and undersampling
2. **Model Training** - Model training with MLflow tracking and artifact logging
3. **Batch Inference** - Distributed inference using Ray for scalability
4. **Redshift Load** - Loading results into data warehouse for analytics

## File Structure

```
sagemaker-pipelines-blueprints/
├── README.md                    # This file
├── pipeline.py                  # Main pipeline orchestration script
├── data_preparation.py          # Data preparation step
├── train.py                     # Model training step
├── inference.py                 # Batch inference step
└── s3_to_redshift.py           # Redshift loading step
```

## Getting Started

### Prerequisites

- AWS Account with SageMaker, S3, MLflow, and (optionally) Redshift access
- Docker container images with required dependencies
- IAM roles with appropriate permissions

### Quick Start

1. **Customize Configuration** - Update the configuration section in `pipeline.py`:

```python
# AWS Configuration
region = "us-east-1"
role = "arn:aws:iam::YOUR-ACCOUNT:role/AmazonSageMaker-ExecutionRole"
bucket = "your-s3-bucket"

# MLflow Configuration
mlflow_tracking_uri = "arn:aws:sagemaker:region:account:mlflow-tracking-server/server-name"
```

2. **Define Your Features** - Update feature definitions in each script:

```python
LABEL = "your_target_column"
NUM_FEATURES = ["feature_1", "feature_2"]
CAT_FEATURES = ["category_1", "category_2"]
```

3. **Run the Pipeline**:

```bash
python pipeline.py
```

## Detailed Component Guide

### 1. Pipeline Orchestration (`pipeline.py`)

The main orchestration script that defines and executes the complete pipeline.

**Key Features:**
- Parameterized pipeline for easy experimentation
- Step-by-step dependency management
- Caching support for faster reruns
- MLflow integration for experiment tracking

**Customization Points:**
- Container image URIs
- Instance types and counts
- Hyperparameters
- S3 paths and prefixes

**Example Usage:**
```python
# Override parameters at execution time
execution = pipeline.start(
    parameters={
        "ModelVersion": "v2.0",
        "Iterations": 1000,
        "LearningRate": 0.05,
    }
)
```

### 2. Data Preparation (`data_preparation.py`)

Handles data loading, preprocessing, and train-test splitting.

**Key Features:**
- Optional AWS Glue job triggering for large-scale ETL
- Stratified train-test split
- Class imbalance handling via undersampling
- Metadata tracking

**Customization Points:**
- Data loading logic (S3, local, databases)
- Feature engineering
- Sampling strategies
- Train-test split ratios

**Example Usage:**
```bash
python data_preparation.py \
    --bucket my-bucket \
    --test-size 0.3 \
    --sampling-ratio 0.1 \
    --output-prefix data/prepared
```

### 3. Model Training (`train.py`)

Trains ML models with comprehensive MLflow tracking.

**Key Features:**
- Multiple model support (CatBoost, XGBoost, LightGBM, etc.)
- Automatic metric calculation and logging
- SHAP value generation for explainability
- Model artifact versioning

**Customization Points:**
- Model type and architecture
- Hyperparameters
- Evaluation metrics
- Feature engineering in training loop

**Example Model Replacement:**
```python
# Replace CatBoost with XGBoost
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=args.iterations,
    learning_rate=args.learning_rate,
    max_depth=args.depth,
)
model.fit(X_train, y_train)
```

### 4. Batch Inference (`inference.py`)

Distributed batch inference using Ray for processing large datasets.

**Key Features:**
- Ray-based distributed processing
- Automatic model loading from previous step
- SHAP value calculation for predictions
- Glue crawler integration for data cataloging

**Customization Points:**
- Batch size and parallelism
- Postprocessing logic
- Output schema and format
- Business rules and transformations

**Example Usage:**
```bash
python inference.py \
    --data-path s3://bucket/input-data \
    --model-path s3://bucket/model.cb \
    --output-path s3://bucket/predictions \
    --batch-size 10000 \
    --num-cpus 32
```

### 5. Redshift Load (`s3_to_redshift.py`)

Loads inference results from S3 into Amazon Redshift.

**Key Features:**
- Secure credential management via SSM Parameter Store
- Efficient COPY command for bulk loading
- Automatic table creation and optimization
- Data validation and verification

**Customization Points:**
- Connection parameters
- Table schema and column types
- Data transformations
- Error handling and retries

**Example Usage:**
```bash
python s3_to_redshift.py \
    --s3-path s3://bucket/predictions \
    --redshift-schema my_schema \
    --redshift-table predictions \
    --redshift-iam-role arn:aws:iam::account:role/RedshiftRole
```

## Pipeline Parameters

The pipeline supports the following parameters that can be overridden at execution time:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ModelVersion` | String | `v1.0` | Model version identifier |
| `DateFilter` | String | `date = DATE '2025-01-01'` | Date filter for data |
| `TestSize` | Float | `0.3` | Proportion of test data |
| `SamplingRatio` | Float | `0.1` | Undersampling ratio |
| `Iterations` | Integer | `500` | Model training iterations |
| `LearningRate` | Float | `0.1` | Model learning rate |
| `Depth` | Integer | `6` | Model depth |
| `EarlyStopRounds` | Integer | `50` | Early stopping rounds |
| `InstanceType` | String | `ml.m6i.8xlarge` | Compute instance type |
| `InstanceCount` | Integer | `4` | Number of instances |
| `BatchSize` | Integer | `10000` | Inference batch size |

## MLflow Integration

The pipeline integrates with MLflow for comprehensive experiment tracking:

- **Runs**: Each training execution creates a unique MLflow run
- **Metrics**: Accuracy, precision, recall, F1, ROC-AUC, Gini
- **Parameters**: All hyperparameters and configuration values
- **Artifacts**: Models, SHAP values, plots, metadata
- **Models**: Versioned model registry for deployment

## Docker Container Requirements

Your container images should include:

```dockerfile
# Base dependencies
pip install sagemaker-training
pip install sagemaker-mlflow

# Model training dependencies
pip install catboost  # or xgboost, lightgbm, etc.
pip install scikit-learn
pip install pandas numpy

# MLflow tracking
pip install mlflow>=2.0

# Data processing
pip install awswrangler
pip install imbalanced-learn

# Inference dependencies
pip install ray[default]
pip install boto3

# Redshift dependencies
pip install redshift-connector
```

## IAM Permissions

Required IAM permissions for the SageMaker execution role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket/*",
        "arn:aws:s3:::your-bucket"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateProcessingJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:DescribeProcessingJob"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "glue:StartCrawler",
        "glue:GetCrawler",
        "glue:StartJobRun",
        "glue:GetJobRun"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
```

## Best Practices

1. **Version Control**: Use git to track changes to your pipeline code
2. **Testing**: Test each component individually before running the full pipeline
3. **Monitoring**: Set up CloudWatch alarms for pipeline failures
4. **Cost Optimization**: Use appropriate instance types and enable caching
5. **Security**: Store credentials in SSM Parameter Store or Secrets Manager
6. **Logging**: Enable comprehensive logging for debugging
7. **Data Quality**: Validate data at each step of the pipeline

## Troubleshooting

### Common Issues

**Pipeline Step Fails**
- Check CloudWatch logs for detailed error messages
- Verify IAM permissions are correctly configured
- Ensure Docker containers have all required dependencies

**MLflow Tracking Issues**
- Verify MLflow tracking URI is correct
- Check network connectivity to MLflow server
- Ensure MLflow server version is compatible

**Out of Memory Errors**
- Reduce batch size for inference
- Use larger instance types
- Implement data streaming/chunking

**S3 Access Denied**
- Verify IAM role has S3 permissions
- Check bucket policies
- Ensure correct bucket name and region

## Advanced Customization

### Adding Custom Steps

To add a new step to the pipeline:

1. Create a new Python script (e.g., `model_evaluation.py`)
2. Define a ProcessingStep in `pipeline.py`:

```python
evaluation = ProcessingStep(
    name="ModelEvaluation",
    processor=ScriptProcessor(...),
    code="model_evaluation.py",
    depends_on=[training],
)
```

3. Add the step to the pipeline:

```python
pipeline = Pipeline(
    steps=[data_prep, training, evaluation, inference, load]
)
```

### Using Different Model Types

To use a different model framework:

1. Update `train.py` imports and model initialization
2. Modify MLflow logging (e.g., `mlflow.xgboost.log_model()`)
3. Update container image with required dependencies
4. Adjust hyperparameters as needed

### Custom Feature Engineering

Add custom feature engineering in `data_preparation.py`:

```python
def custom_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add your custom feature engineering logic here."""
    df['new_feature'] = df['feature_1'] * df['feature_2']
    return df
```

## Resources

- [SageMaker Pipelines Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Ray Documentation](https://docs.ray.io/)
- [AWS Data Wrangler](https://aws-sdk-pandas.readthedocs.io/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review AWS documentation
3. Check CloudWatch logs for detailed error messages

## License

These blueprints are provided as examples for educational purposes.
Customize as needed for your specific use case.


