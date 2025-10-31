#!/usr/bin/env python3
"""
pipeline.py

Generic blueprint for SageMaker SDK Pipeline orchestration.
This script demonstrates how to build an end-to-end ML pipeline with:
- Data preparation (with optional Glue ETL)
- Model training with MLflow tracking
- Batch inference with distributed processing
- Loading results into data warehouse (Redshift)

Customize this template for your specific use case by:
1. Updating configuration values (bucket, role, paths, etc.)
2. Adjusting hyperparameters for your model
3. Modifying instance types and counts based on your workload
4. Adding/removing pipeline steps as needed
"""

from datetime import datetime

import boto3
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.network import NetworkConfig
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

# ============================================================================
# Configuration
# ============================================================================
# AWS Configuration
region = "us-east-1"
role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
bucket = "my-s3-bucket"
base_prefix = "data"
job_id = f"pipe-{datetime.now().strftime('%Y%m%d%H%M')}"

# MLflow Configuration
mlflow_tracking_uri = (
    "arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/my-mlflow-server"
)
mlflow_bucket = "sagemaker-us-east-1-123456789012"
mlflow_experiment_path = "mlflow-experiments/1"

# Create SageMaker session
sm_session = sagemaker.Session(boto3.Session(region_name=region))

# Network configuration (adjust security groups and subnets for your VPC)
network_config = NetworkConfig(
    security_group_ids=["sg-xxxxxxxxxxxxxxxxx"],
    subnets=[
        "subnet-xxxxxxxxxxxxxxxxx",
        "subnet-yyyyyyyyyyyyyyyyy",
        "subnet-zzzzzzzzzzzzzzzzz",
    ],
)

# Cache configuration for pipeline steps
cache_config = CacheConfig(
    enable_caching=True,
    expire_after="1 day",
)

# ============================================================================
# Pipeline Parameters
# ============================================================================
# Define parameters that can be overridden at pipeline execution time
model_version = ParameterString(name="ModelVersion", default_value="v1.0")
date_filter = ParameterString(
    name="DateFilter", default_value="reporting_date = DATE '2025-01-01'"
)
test_size = ParameterFloat(name="TestSize", default_value=0.3)
sampling_ratio = ParameterFloat(name="SamplingRatio", default_value=0.1)

# Model hyperparameters
iterations = ParameterInteger(name="Iterations", default_value=500)
learning_rate = ParameterFloat(name="LearningRate", default_value=0.1)
depth = ParameterInteger(name="Depth", default_value=6)
early_stop_rounds = ParameterInteger(name="EarlyStopRounds", default_value=50)

# Compute configuration
instance_type = ParameterString(name="InstanceType", default_value="ml.m6i.8xlarge")
instance_count = ParameterInteger(name="InstanceCount", default_value=4)
batch_size = ParameterInteger(name="BatchSize", default_value=10000)

# ============================================================================
# Step 1: Data Preparation
# ============================================================================
# This step uses AWS Glue (via a processing script) to handle large-scale ETL
# and dataset preparation. The script can optionally trigger a Glue job for
# feature engineering, then performs train-test split and optional undersampling.

data_prep = ProcessingStep(
    name="DataPreparation",
    processor=ScriptProcessor(
        command=["python3"],
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-ml-container:latest",
        role=role,
        instance_count=1,
        instance_type="ml.m6i.16xlarge",
        sagemaker_session=sm_session,
    ),
    code="data_preparation.py",
    job_arguments=[
        "--bucket",
        bucket,
        "--base_prefix",
        base_prefix,
        "--out_prefix",
        f"{base_prefix}/ml-dataset",
        "--crawler_name",
        "ML_DATASET_CRAWLER",
        "--custom_date_filter",
        date_filter,
        "--run-glue-job",  # Optional: remove if you don't need Glue ETL
        "--test-size",
        test_size.to_string(),
        "--sampling-ratio",
        sampling_ratio.to_string(),
        "--output-prefix",
        f"{base_prefix}/ml-dataset-stratified/{job_id}",
        "--job-id",
        job_id,
    ],
    cache_config=cache_config,
)

# ============================================================================
# Step 2: Model Training
# ============================================================================
# This step trains the ML model using the prepared data from Step 1.
# It automatically logs metrics, parameters, and artifacts to MLflow.

# Define the training estimator
estimator = Estimator(
    image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-ml-container:latest",
    role=role,
    instance_count=1,
    instance_type="ml.m6i.32xlarge",
    sagemaker_session=sm_session,
    entry_point="train.py",
    source_dir=".",
    environment={"MLFLOW_TRACKING_URI": mlflow_tracking_uri},
    output_path=f"s3://{bucket}/{base_prefix}/ml-artifacts",
)

# Set hyperparameters that will be passed to train.py
estimator.set_hyperparameters(
    model_version=model_version,
    iterations=iterations,
    learning_rate=learning_rate,
    depth=depth,
    early_stopping_rounds=early_stop_rounds,
)

# Create the training step
training = TrainingStep(
    name="ModelTraining",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=f"s3://{bucket}/{base_prefix}/ml-dataset-stratified/{job_id}/train",
            content_type="application/x-parquet",
        ),
        "test": TrainingInput(
            s3_data=f"s3://{bucket}/{base_prefix}/ml-dataset-stratified/{job_id}/test",
            content_type="application/x-parquet",
        ),
    },
    depends_on=[data_prep],
    cache_config=cache_config,
)

# ============================================================================
# Step 3: Batch Inference
# ============================================================================
# This step performs distributed batch inference using the trained model.
# It processes large datasets in parallel and outputs predictions to S3.

inference = ProcessingStep(
    name="BatchInference",
    processor=ScriptProcessor(
        command=["python3"],
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-inference-container:latest",
        role=role,
        instance_count=4,
        instance_type="ml.m6i.8xlarge",
        volume_size_in_gb=100,
        sagemaker_session=sm_session,
        env={"MLFLOW_TRACKING_URI": mlflow_tracking_uri},
    ),
    code="inference.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/{base_prefix}/ml-dataset",
            destination="/opt/ml/processing/input",
        ),
        ProcessingInput(
            source=training.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model-artifacts",
        ),
        ProcessingInput(
            source=Join(
                on="",
                values=[
                    f"s3://{bucket}/{base_prefix}/ml-artifacts/",
                    training.properties.TrainingJobName,
                    "/output/",
                ],
            ),
            destination="/opt/ml/processing/mlflow",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="inference_results",
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/{base_prefix}/outputs/inference/{job_id}",
        ),
        ProcessingOutput(
            output_name="inference_metadata",
            source="/opt/ml/processing/output_metadata",
            destination=f"s3://{bucket}/{base_prefix}/outputs/inference/{job_id}/metadata",
        ),
    ],
    job_arguments=[
        "--data-path",
        "/opt/ml/processing/input",
        "--mlflow-metadata",
        "/opt/ml/processing/mlflow",
        "--model-version",
        model_version,
        "--output-path",
        "/opt/ml/processing/output",
        "--region",
        region,
        "--bucket",
        bucket,
        "--glue-crawler",
        "BATCH_INFERENCE_CRAWLER",
        "--glue-database",
        "my_database",
        "--glue-table",
        "model_scores",
    ],
    depends_on=[training],
)

# ============================================================================
# Step 4: Load to Redshift
# ============================================================================
# This step loads the inference results from S3 into Amazon Redshift for
# downstream analytics and reporting.

load = ProcessingStep(
    name="RedshiftLoad",
    processor=ScriptProcessor(
        command=["python3"],
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-inference-container:latest",
        role=role,
        instance_count=1,
        instance_type="ml.m6i.large",
        sagemaker_session=sm_session,
        env={"AWS_DEFAULT_REGION": region},
        network_config=network_config,
    ),
    code="s3_to_redshift.py",
    inputs=[
        ProcessingInput(
            source=inference.properties.ProcessingOutputConfig.Outputs[
                "inference_metadata"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/metadata",
        )
    ],
    job_arguments=[
        "--metadata-path",
        "/opt/ml/processing/metadata/metadata.json",
        "--redshift-schema",
        "my_schema",
        "--redshift-table",
        "model_scores",
        "--redshift-iam-role",
        "arn:aws:iam::123456789012:role/RedshiftCopyRole",
        "--job-name",
        f"redshift-load-job-{job_id}",
    ],
    depends_on=[inference],
)

# ============================================================================
# Create and Execute Pipeline
# ============================================================================
# Combine all steps into a single pipeline

pipeline = Pipeline(
    name=f"ml-training-pipeline-{job_id}",
    parameters=[
        model_version,
        date_filter,
        test_size,
        sampling_ratio,
        iterations,
        learning_rate,
        depth,
        early_stop_rounds,
        instance_type,
        instance_count,
        batch_size,
    ],
    steps=[data_prep, training, inference, load],
    sagemaker_session=sm_session,
)

# Create or update the pipeline
pipeline.upsert(role_arn=role)

# Start pipeline execution with default parameters
# You can override parameters by passing them to pipeline.start()
execution = pipeline.start()

print(f"✅ Pipeline started: {execution.arn}")
print(f"Pipeline execution name: {execution._execution_name}")
print("Monitor execution in SageMaker Studio or via AWS Console")

# Optional: Override parameters at execution time
# execution = pipeline.start(
#     parameters={
#         "ModelVersion": "v2.0",
#         "Iterations": 1000,
#         "LearningRate": 0.05,
#     }
# )
