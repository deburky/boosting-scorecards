#!/usr/bin/env python3
"""
sagemaker_inference.py

Launches a SageMaker processing job for batch inference using the CatBoost model.
"""

import argparse
import os
import pathlib
import uuid
from datetime import datetime

import boto3
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
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
INSTANCE_TYPE = (
    "ml.m6i.8xlarge"  # Default instance type for inference - 8xlarge for larger memory
)
VOLUME_SIZE = 100

# S3 paths
S3_DATA_PATH = "s3://my-s3-bucket/data/ml-dataset/"
# Use the model from our latest successful training job
S3_MODEL_URI = "s3://sagemaker-us-east-1-497487485332/clf-model-v1-1-output/clf-model-v1-1-10071345-f70e9893/output/model.tar.gz"
S3_OUTPUT_PATH = "s3://my-s3-bucket/outputs/inference"

# Generate unique job ID
JOB_ID = f"{datetime.now().strftime('%m%d%H%M')}-{str(uuid.uuid4())[:8]}"


def create_source_dir():
    """
    Create a source directory with the inference script.

    Returns:
        Path to the source directory
    """
    source_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "source_inference"
    )
    os.makedirs(source_dir, exist_ok=True)

    # Simply copy the inference script to the source directory without modification
    inference_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "inference.py"
    )
    source_inference_script = os.path.join(source_dir, "inference.py")

    inference_content = pathlib.Path(inference_script).read_text(encoding="utf-8")
    # Write the inference script as is
    with open(source_inference_script, "w", encoding="utf-8") as f:
        f.write(inference_content)

    print(f"Created source directory: {source_dir}")
    return source_dir


def run_sagemaker_processing_job(
    dry_run=False,
    model_version="v1.1",
    instance_type=INSTANCE_TYPE,
    sample_size=10000,
    dynamic_thresholds=False,
):
    """Run a SageMaker processing job."""
    print(f"CatBoost Batch Inference in SageMaker (Job ID: {JOB_ID})\n")

    if dry_run:
        print("DRY RUN MODE - No SageMaker job will be launched\n")
        print(f"Region: {REGION}")
        print(f"Container URI: {CONTAINER_URI}")
        print(f"Instance Type: {instance_type}")
        print(f"MLflow Tracking URI: {TRACKING_SERVER_ARN}")
        print(f"Sample Size: {sample_size}")
        print(f"Dynamic Thresholds: {dynamic_thresholds}")
        return

    boto_session = boto3.Session(region_name=REGION)
    sagemaker_session = Session(boto_session=boto_session)
    source_dir = create_source_dir()

    # Replace dots with hyphens in model_version for SageMaker job name compatibility
    safe_model_version = model_version.replace(".", "-")

    # Create a ScriptProcessor for running inference
    script_processor = ScriptProcessor(
        image_uri=CONTAINER_URI,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        volume_size_in_gb=VOLUME_SIZE,
        max_runtime_in_seconds=24 * 60 * 60,  # 24 hours
        role=SAGEMAKER_ROLE,
        env={"MLFLOW_TRACKING_URI": TRACKING_SERVER_ARN},
        sagemaker_session=sagemaker_session,
    )

    # Define input and output data channels
    inputs = [
        ProcessingInput(
            source=S3_DATA_PATH,
            destination="/opt/ml/processing/input",
            s3_data_distribution_type="FullyReplicated",
        ),
        ProcessingInput(
            source=S3_MODEL_URI,
            destination="/opt/ml/processing/model/model.tar.gz",
            s3_data_distribution_type="FullyReplicated",
        ),
        ProcessingInput(
            source=source_dir,
            destination="/opt/ml/processing/code",
            s3_data_distribution_type="FullyReplicated",
        ),
    ]

    outputs = [
        ProcessingOutput(
            output_name="inference_results",
            source="/opt/ml/processing/output",
            destination=f"{S3_OUTPUT_PATH}/{model_version}/{JOB_ID}",
        )
    ]

    # Prepare arguments for the processing script
    arguments = [
        "--data-path",
        "/opt/ml/processing/input",
        "--model-path",
        "/opt/ml/processing/model/model.tar.gz",
        "--output-path",
        "/opt/ml/processing/output",
        "--batch-size",
        "1000",
        "--sample-size",
        str(sample_size),
    ]

    # Add dynamic thresholds flag if specified
    if dynamic_thresholds:
        arguments.append("--dynamic-thresholds")

    job_name = f"clf-inference-{safe_model_version}-{JOB_ID.replace('_', '-')}"
    print(f"\nStarting SageMaker processing job: {job_name}")
    print(f"Input data path: {S3_DATA_PATH}")
    print(f"Model URI: {S3_MODEL_URI}")
    print(f"Output path: {outputs[0].destination}")
    print(f"Sample size: {sample_size}")
    print(f"Dynamic thresholds: {dynamic_thresholds}")

    # Run the processing job
    script_processor.run(
        code=os.path.join(source_dir, "inference.py"),
        inputs=inputs,
        outputs=outputs,
        arguments=arguments,
        job_name=job_name,
        wait=False,
        logs=False,
    )

    print("\nSageMaker processing job submitted!")
    print(f"Job name: {job_name}")
    print(
        f"Check job status in SageMaker console: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/processing-jobs/{job_name}"
    )
    print(f"Results will be saved to: {outputs[0].destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference for CatBoost model in SageMaker"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="No SageMaker job will be launched"
    )
    parser.add_argument(
        "--model-version", type=str, default="v1.1", help="Model version to use"
    )
    parser.add_argument(
        "--sample-size", type=int, default=10000, help="Number of records to process"
    )
    parser.add_argument(
        "--dynamic-thresholds",
        action="store_true",
        help="Calculate thresholds dynamically from data",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default=INSTANCE_TYPE,
        help="SageMaker instance type",
    )

    args = parser.parse_args()

    run_sagemaker_processing_job(
        dry_run=args.dry_run,
        model_version=args.model_version,
        instance_type=args.instance_type,
        sample_size=args.sample_size,
        dynamic_thresholds=args.dynamic_thresholds,
    )
