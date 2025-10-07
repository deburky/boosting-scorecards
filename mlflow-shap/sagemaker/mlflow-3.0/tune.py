#!/usr/bin/env python3
"""
tune.py

Script to run hyperparameter tuning for CatBoost model using SageMaker Automatic Model Tuning (AMT).
"""

import argparse
import os
import pathlib
import uuid
from datetime import datetime

import boto3
from sagemaker.estimator import Estimator
from sagemaker.session import Session
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter

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
INSTANCE_TYPE = "ml.m6i.32xlarge"  # Default instance type for tuning
VOLUME_SIZE = 100

# S3 data paths
S3_DATA_PATH = "s3://my-s3-bucket/data/ml-dataset"

# Generate unique job ID
JOB_ID = f"{datetime.now().strftime('%m%d%H%M')}-{str(uuid.uuid4())[:8]}"


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


def run_hyperparameter_tuning(
    model_version="v1.1",
    max_jobs=10,
    max_parallel_jobs=2,
    wait=True,
    instance_type=INSTANCE_TYPE,
):  # sourcery skip: extract-method
    """
    Run hyperparameter tuning for CatBoost model.

    Args:
        model_version: Model version tag
        max_jobs: Maximum number of training jobs to run
        max_parallel_jobs: Maximum number of parallel training jobs
        wait: Whether to wait for the tuning job to complete
        instance_type: SageMaker instance type to use

    Returns:
        SageMaker HyperparameterTuner object
    """
    print(f"\nStarting hyperparameter tuning for CatBoost model {model_version}...")
    print(f"Job ID: {JOB_ID}")

    # Create a boto3 session
    boto_session = boto3.Session(region_name=REGION)

    # Create SageMaker session
    sagemaker_session = Session(boto_session=boto_session)

    # Create source directory with training script
    source_dir = create_source_dir()

    # Base hyperparameters (non-tunable)
    hyperparameters = {
        "model_version": model_version,
        "sampling_ratio": 0.1,  # Use 10% of negative class samples
    }

    # Create estimator
    estimator = Estimator(
        image_uri=CONTAINER_URI,
        role=SAGEMAKER_ROLE,
        instance_count=1,
        instance_type=instance_type,
        volume_size=VOLUME_SIZE,
        max_run=24 * 60 * 60,  # 24 hours
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters,
        environment={"MLFLOW_TRACKING_URI": TRACKING_SERVER_ARN},
        entry_point="train.py",  # Specify the entry point
        source_dir=source_dir,  # Specify the source directory
    )

    # Define hyperparameter ranges to tune
    hyperparameter_ranges = {
        "iterations": IntegerParameter(50, 500),
        "learning_rate": ContinuousParameter(0.01, 0.3, scaling_type="Logarithmic"),
        "depth": IntegerParameter(4, 10),
        "l2_leaf_reg": ContinuousParameter(0.1, 10.0, scaling_type="Logarithmic"),
        "early_stopping_rounds": IntegerParameter(10, 50),
    }

    # Define objective metric
    objective_metric_name = "roc_auc"
    objective_type = "Maximize"  # Higher ROC AUC is better

    # Define regex for parsing metrics from logs
    metric_definitions = [
        {"Name": "roc_auc", "Regex": "ROC AUC: ([0-9\\.]+)"},
        {"Name": "accuracy", "Regex": "Test accuracy: ([0-9\\.]+)"},
        {"Name": "f1", "Regex": "F1 Score: ([0-9\\.]+)"},
        {"Name": "precision", "Regex": "Precision: ([0-9\\.]+)"},
        {"Name": "recall", "Regex": "Recall: ([0-9\\.]+)"},
    ]

    # Replace dots with hyphens in model_version for SageMaker job name compatibility
    safe_model_version = model_version.replace(".", "-")

    # Create hyperparameter tuner
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=objective_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=metric_definitions,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        objective_type=objective_type,
        base_tuning_job_name=f"clf-tuning-{safe_model_version}",
    )

    # Set up the input data channel
    input_data = {"train": S3_DATA_PATH}

    # Run hyperparameter tuning job
    tuner.fit(
        inputs=input_data,
        job_name=f"clf-tune-{safe_model_version}-{JOB_ID.replace('_', '-')}",
        wait=wait,
        logs="All" if wait else None,
    )

    print("\nHyperparameter tuning job submitted.")

    # If wait=True, show the best hyperparameters
    if wait:
        best_training_job = tuner.best_training_job()
        print(f"\nBest training job: {best_training_job}")

        best_job_details = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=best_training_job
        )

        print("\nBest hyperparameters:")
        for param, value in best_job_details["HyperParameters"].items():
            if not param.startswith("sagemaker_"):
                print(f"  {param}: {value}")

        # Get the metrics
        for metric in best_job_details.get("FinalMetricDataList", []):
            print(f"  {metric['MetricName']}: {metric['Value']}")

        # Save best hyperparameters to a file
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"best_hyperparameters_{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Best hyperparameters for model version {model_version}:\n\n")
            for param, value in best_job_details["HyperParameters"].items():
                if not param.startswith("sagemaker_"):
                    f.write(f"{param}: {value}\n")

            f.write("\nMetrics:\n")
            for metric in best_job_details.get("FinalMetricDataList", []):
                f.write(f"{metric['MetricName']}: {metric['Value']}\n")

        print(f"\nBest hyperparameters saved to {output_path}")

    return tuner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning for CatBoost model"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.1",
        help="Model version",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=10,
        help="Maximum number of training jobs to run",
    )
    parser.add_argument(
        "--max-parallel-jobs",
        type=int,
        default=2,
        help="Maximum number of parallel training jobs",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for the tuning job to complete",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default=INSTANCE_TYPE,
        help="SageMaker instance type to use",
    )

    args = parser.parse_args()

    run_hyperparameter_tuning(
        model_version=args.model_version,
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel_jobs,
        wait=not args.no_wait,
        instance_type=args.instance_type,
    )
