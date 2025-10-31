#!/usr/bin/env python3
"""
s3_to_redshift.py

Generic blueprint for loading data from S3 to Amazon Redshift.
This script demonstrates how to:
- Connect to Redshift using credentials from AWS SSM Parameter Store
- Load data from S3 using Redshift COPY command
- Handle table creation, data loading, and vacuuming
- Verify data loaded successfully

Customize this template by:
1. Updating connection parameters for your Redshift cluster
2. Adjusting table schema and column types
3. Modifying error handling and retry logic
4. Adding custom data transformations or validations
"""

import argparse
import json
import logging
import sys
import time

import awswrangler as wr
import boto3
import botocore
import redshift_connector

# Configure AWS Data Wrangler
wr.engine.set("python")
wr.config.botocore_config = botocore.config.Config(
    retries={"max_attempts": 10}, connect_timeout=20, max_pool_connections=100
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Argument Parsing
# ============================================================================
def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transfer data from S3 to Redshift")

    # S3 parameters
    parser.add_argument(
        "--s3-path",
        type=str,
        required=True,
        help="S3 path to the data (e.g., s3://bucket/prefix/)",
    )

    # Glue catalog parameters (optional - for schema discovery)
    parser.add_argument(
        "--glue-database",
        type=str,
        default=None,
        help="Glue database name (optional)",
    )
    parser.add_argument(
        "--glue-table",
        type=str,
        default=None,
        help="Glue table name (optional)",
    )

    # Redshift target parameters
    parser.add_argument(
        "--redshift-schema",
        type=str,
        required=True,
        help="Redshift schema name",
    )
    parser.add_argument(
        "--redshift-table",
        type=str,
        required=True,
        help="Redshift table name",
    )
    parser.add_argument(
        "--redshift-iam-role",
        type=str,
        required=True,
        help="IAM role ARN for Redshift to access S3",
    )

    # Redshift connection parameters
    parser.add_argument(
        "--redshift-host",
        type=str,
        default="my-redshift-cluster.region.redshift.amazonaws.com",
        help="Redshift host",
    )
    parser.add_argument(
        "--redshift-database",
        type=str,
        default="my_database",
        help="Redshift database",
    )
    parser.add_argument(
        "--redshift-port",
        type=int,
        default=5439,
        help="Redshift port",
    )
    parser.add_argument(
        "--redshift-user",
        type=str,
        default="admin",
        help="Redshift user",
    )
    parser.add_argument(
        "--redshift-password-path",
        type=str,
        default="/PROD/REDSHIFT_PASSWORD",
        help="SSM parameter path for Redshift password",
    )

    # Optional metadata path (from previous inference step)
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Path to metadata.json from inference step",
    )

    # Optional job name
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Unique job name for tracking",
    )

    return parser.parse_args()


# ============================================================================
# Main Transfer Function
# ============================================================================
def write_to_redshift(args):
    """
    Write data from S3 to Redshift with comprehensive logging.

    Args:
        args: Parsed command-line arguments
    """
    logger.info("=" * 80)
    logger.info("Starting S3 to Redshift transfer")
    logger.info("Source: %s", args.s3_path)
    logger.info("Destination: %s.%s", args.redshift_schema, args.redshift_table)
    logger.info("=" * 80)

    # Set up boto3 session
    boto3.setup_default_session(region_name="us-east-1")
    session = boto3.Session(region_name="us-east-1")

    # Get Redshift password from SSM Parameter Store
    ssm_client = boto3.client("ssm")

    try:
        logger.info("Retrieving password from SSM: %s", args.redshift_password_path)
        password = ssm_client.get_parameter(
            Name=args.redshift_password_path, WithDecryption=True
        )["Parameter"]["Value"]
        logger.info("Password retrieved successfully")
    except Exception as e:
        logger.error("Error retrieving password from SSM: %s", str(e))
        sys.exit(1)

    # Connect to Redshift
    try:
        logger.info(
            "Connecting to Redshift at %s:%s", args.redshift_host, args.redshift_port
        )
        con = redshift_connector.connect(
            host=args.redshift_host,
            database=args.redshift_database,
            port=args.redshift_port,
            user=args.redshift_user,
            password=password,
        )
        logger.info("Connected to Redshift successfully")
    except Exception as e:
        logger.error("Error connecting to Redshift: %s", str(e))
        sys.exit(1)

    try:
        total_start = time.perf_counter()

        # Drop existing table if it exists
        logger.info(
            "Dropping table %s.%s if exists", args.redshift_schema, args.redshift_table
        )
        step_start = time.perf_counter()

        with con.cursor() as cursor:
            cursor.execute(
                f"DROP TABLE IF EXISTS {args.redshift_schema}.{args.redshift_table}"
            )
        con.commit()

        elapsed = time.perf_counter() - step_start
        logger.info("Table dropped in %.1f seconds", elapsed)

        # Load data from S3 to Redshift using AWS Data Wrangler
        logger.info("Starting data copy from S3 to Redshift")
        step_start = time.perf_counter()

        try:
            # Log current execution role
            sts_client = session.client("sts")
            caller_identity = sts_client.get_caller_identity()
            current_role_arn = caller_identity["Arn"]
            logger.info("Current execution role: %s", current_role_arn)

            # Copy data from S3 to Redshift
            logger.info("Copying data from %s", args.s3_path)

            wr.redshift.copy_from_files(
                path=args.s3_path,
                con=con,
                schema=args.redshift_schema,
                table=args.redshift_table,
                iam_role=args.redshift_iam_role,
                diststyle="EVEN",
                data_format="parquet",
                mode="overwrite",
                varchar_lengths_default=1000,
            )

            logger.info(
                "Successfully copied data to %s.%s",
                args.redshift_schema,
                args.redshift_table,
            )

        except Exception as e:
            logger.error("Error transferring data: %s", str(e))
            logger.error("Data transfer failed")
            sys.exit(1)

        elapsed = time.perf_counter() - step_start
        mins, secs = divmod(elapsed, 60)
        logger.info("Data copied to Redshift in %d min %.1f sec", int(mins), secs)

        # Verify data was loaded successfully
        logger.info("Verifying data was loaded successfully")
        with con.cursor() as cursor:
            try:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {args.redshift_schema}.{args.redshift_table}"
                )
                count = cursor.fetchone()[0]
                logger.info(
                    "✅ Successfully loaded %s rows into %s.%s",
                    f"{count:,}",
                    args.redshift_schema,
                    args.redshift_table,
                )
            except Exception as e:
                logger.error("Error verifying data: %s", str(e))

        # Commit and close connection
        con.commit()
        con.close()

        # Vacuum table (optional but recommended)
        logger.info("Vacuuming table to optimize storage")
        step_start = time.perf_counter()

        con = redshift_connector.connect(
            host=args.redshift_host,
            database=args.redshift_database,
            port=args.redshift_port,
            user=args.redshift_user,
            password=password,
        )
        con.autocommit = True

        with con.cursor() as cursor:
            cursor.execute(f"VACUUM FULL {args.redshift_schema}.{args.redshift_table}")

        elapsed = time.perf_counter() - step_start
        mins, secs = divmod(elapsed, 60)
        logger.info("Table vacuumed in %d min %.1f sec", int(mins), secs)

        # Calculate total time
        total_elapsed = time.perf_counter() - total_start
        mins, secs = divmod(total_elapsed, 60)
        logger.info("=" * 80)
        logger.info("✅ Redshift load complete in %d min %.1f sec", int(mins), secs)
        logger.info("=" * 80)

    except Exception as e:
        logger.error("Error during data transfer: %s", str(e))
        sys.exit(1)
    finally:
        con.close()


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main entry point."""
    args = get_arguments()

    # If metadata path is provided, load it and potentially override S3 path
    if args.metadata_path:
        try:
            logger.info("Loading metadata from %s", args.metadata_path)
            with open(args.metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)

            logger.info("Metadata loaded: %s", metadata)

            # Use S3 output path from metadata if available
            if "s3_output_path" in metadata:
                args.s3_path = metadata["s3_output_path"]
                logger.info("Using S3 path from metadata: %s", args.s3_path)

            # Use Glue table from metadata if available
            if "glue_table" in metadata and not args.glue_table:
                args.glue_table = metadata["glue_table"]
                logger.info("Using Glue table from metadata: %s", args.glue_table)

        except Exception as e:
            logger.warning("Failed to load metadata: %s", str(e))
            logger.warning("Continuing with command-line arguments")

    # Validate S3 path
    if not args.s3_path or not args.s3_path.startswith("s3://"):
        logger.error("Invalid S3 path: %s", args.s3_path)
        sys.exit(1)

    # Execute transfer
    write_to_redshift(args)

    logger.info("S3 to Redshift transfer completed successfully")


if __name__ == "__main__":
    main()
