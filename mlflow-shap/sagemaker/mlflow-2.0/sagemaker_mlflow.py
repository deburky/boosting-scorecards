#!/usr/bin/env python3
"""
sagemaker_mlflow.py

Launches a SageMaker training job for the MLflow SHAP example.
No training data is passed (synthetic data generated inside train.py).
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
    "arn:aws:sagemaker:us-east-1:497487485332:mlflow-tracking-server/sagemaker-mlflow"
)
CONTAINER_URI = (
    "497487485332.dkr.ecr.us-east-1.amazonaws.com/catboost-mlflow-container:latest"
)
INSTANCE_TYPE = "ml.m5.4xlarge"
VOLUME_SIZE = 100

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


def run_sagemaker_job(dry_run=False):
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
    hyperparameters = json_encode_hyperparameters(
        {
            "l2_leaf_reg": 5,
            "min_data_in_leaf": 10,
            "early_stopping_rounds": 10,
        }
    )

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
        output_path=f"s3://{sagemaker_session.default_bucket()}/mlflow-shap-output",
        code_location=f"s3://{sagemaker_session.default_bucket()}/mlflow-shap-code",
    )

    job_name = f"mlflow-shap-{JOB_ID}"
    print(f"\nStarting SageMaker training job: {job_name}")
    estimator.fit(job_name=job_name)

    print("\nSageMaker job complete!")
    print(f"Job name: {job_name}")
    print(f"Model artifacts: {estimator.model_data}")
    print(
        f"\nMLflow UI: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/mlflow"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLflow SHAP example in SageMaker")
    parser.add_argument(
        "--dry-run", action="store_true", help="No SageMaker job will be launched"
    )
    args = parser.parse_args()

    run_sagemaker_job(dry_run=args.dry_run)
