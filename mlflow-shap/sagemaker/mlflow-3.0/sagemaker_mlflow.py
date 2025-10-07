#!/usr/bin/env python3
"""
sagemaker_mlflow.py

Launches a SageMaker training job for the MLflow SHAP example with classification model.
Connects to a dataset stored in S3.
"""

import argparse
import json
import os
import pathlib
import uuid
from datetime import datetime

import boto3
from sagemaker.estimator import Estimator
from sagemaker.session import Session

# SageMaker configuration
REGION = "us-east-1"
SAGEMAKER_ROLE = (
    "arn:aws:iam::497487485332:role/service-role/AmazonSageMaker-ExecutionRole"
)
TRACKING_SERVER_ARN = (
    "arn:aws:sagemaker:us-east-1:497487485332:mlflow-tracking-server/sagemaker-mlflow-3"
)
CONTAINER_URI = (
    "497487485332.dkr.ecr.us-east-1.amazonaws.com/catboost-mlflow-3-container:latest"
)
INSTANCE_TYPE = "ml.m6i.32xlarge"  # Large instance type for processing
VOLUME_SIZE = 100

# S3 data paths
S3_DATA_PATH = "s3://my-s3-bucket/data/ml-dataset"

JOB_ID = f"{datetime.now().strftime('%m%d%H%M')}-{str(uuid.uuid4())[:8]}"


def json_encode_hyperparameters(hyperparameters):
    """Encode hyperparameters into json"""
    return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}


def create_source_dir():
    """
    Create a source directory with the training script.

    Returns:
        Path to the source directory
    """
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")
    os.makedirs(source_dir, exist_ok=True)

    # Copy the training script to the source directory
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    source_train_script = os.path.join(source_dir, "train.py")

    train_content = pathlib.Path(train_script).read_text(encoding="utf-8")
    with open(source_train_script, "w", encoding="utf-8") as f:
        f.write(train_content)

    print(f"Created source directory: {source_dir}")
    return source_dir


def run_sagemaker_job(dry_run=False, model_version="v1.1"):
    """Run a SageMaker job."""
    print(f"MLflow + SHAP in SageMaker (Job ID: {JOB_ID})\n")

    if dry_run:
        print("DRY RUN MODE - No SageMaker job will be launched\n")
        print(f"Region: {REGION}")
        print(f"Container URI: {CONTAINER_URI}")
        print(f"Instance Type: {INSTANCE_TYPE}")
        print(f"MLflow Tracking URI: {TRACKING_SERVER_ARN}")
        return

    boto_session = boto3.Session(region_name=REGION)

    sagemaker_session = Session(boto_session=boto_session)
    source_dir = create_source_dir()
    # Instead of passing the input_data_dir as a hyperparameter,
    # we'll use the SageMaker train channel which maps to SM_CHANNEL_TRAIN
    hyperparameters = {
        "model_version": model_version,
        "iterations": 50,  # Reduced for faster testing (original: 500)
        "learning_rate": 0.1,
        "depth": 6,
        "early_stopping_rounds": 10,  # Reduced for faster testing (original: 50)
        "sampling_ratio": 0.1,
    }

    # Set up the training data channel
    input_data_channel = {"train": S3_DATA_PATH}

    # Replace dots with hyphens in model_version for SageMaker job name compatibility
    safe_model_version = model_version.replace(".", "-")

    estimator = Estimator(
        image_uri=CONTAINER_URI,
        role=SAGEMAKER_ROLE,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        volume_size=VOLUME_SIZE,
        max_run=24 * 60 * 60,
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters,
        environment={"MLFLOW_TRACKING_URI": TRACKING_SERVER_ARN},
        entry_point="train.py",  # Specify the entry point
        source_dir=source_dir,  # Specify the source directory
        output_path=f"s3://{sagemaker_session.default_bucket()}/clf-model-{safe_model_version}-output",
        code_location=f"s3://{sagemaker_session.default_bucket()}/clf-model-{safe_model_version}-code",
        input_mode="FastFile",
    )
    job_name = f"clf-model-{safe_model_version}-{JOB_ID.replace('_', '-')}"
    print(f"\nStarting SageMaker training job: {job_name}")
    print(f"Input data channel: {input_data_channel}")
    estimator.fit(inputs=input_data_channel, job_name=job_name)

    print("\nSageMaker job complete!")
    print(f"Job name: {job_name}")
    print(f"Model artifacts: {estimator.model_data}")
    print(
        f"\nMLflow UI: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/mlflow"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLflow SHAP example in SageMaker")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No SageMaker job will be launched",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.1",
        help="Model version to use",
    )
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default=S3_DATA_PATH,
        help="S3 path for input data",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Use reduced iterations for faster testing",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default=INSTANCE_TYPE,
        help="SageMaker instance type",
    )
    parser.add_argument(
        "--test-instance-type",
        type=str,
        default="ml.m6i.16xlarge",  # More powerful instance for testing (available in SageMaker)
        help="SageMaker instance type to use for test runs",
    )
    args = parser.parse_args()

    if args.input_data_dir != S3_DATA_PATH:
        S3_DATA_PATH = args.input_data_dir

    if args.instance_type != INSTANCE_TYPE:
        INSTANCE_TYPE = args.instance_type

    # Print information about the job
    print(f"Model Version: {args.model_version}")
    print(f"Instance Type: {INSTANCE_TYPE}")
    print(f"Input Data: {S3_DATA_PATH}")

    if args.test_run:
        print("\nRUNNING IN TEST MODE WITH REDUCED ITERATIONS")
        # If test-run is specified, override instance type with test instance type
        INSTANCE_TYPE = args.test_instance_type
        print(f"Using test instance type: {INSTANCE_TYPE}")

    run_sagemaker_job(dry_run=args.dry_run, model_version=args.model_version)
