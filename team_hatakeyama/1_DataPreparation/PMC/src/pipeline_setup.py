import argparse
import os
from dotenv import load_dotenv
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    GoogleCloudOptions,
    WorkerOptions,
)
import secrets
from config import GCPConfig

load_dotenv()


def cli_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default=GCPConfig.REGION)
    parser.add_argument("--start_batch", default=0, type=int, help="Start batch number")
    parser.add_argument("--end_batch", default=10, type=int, help="End batch number")
    parser.add_argument("--gcp_project_id", default=GCPConfig.PROJECT_ID, type=str)
    parser.add_argument(
        "--credidental_path", default=GCPConfig.CREDENTIAL_PATH, type=str
    )
    return parser.parse_known_args(argv)


# Dataflow pipeline setup
def configure_pipeline_options(known_args, pipeline_args, batch_name):
    print(f"⌛️ Setting up pipeline for {batch_name}")
    # Credential path from environment setup
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = known_args.credidental_path
    options = PipelineOptions(pipeline_args)
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.region = known_args.location
    google_cloud_options.project = known_args.gcp_project_id
    google_cloud_options.job_name = f"{GCPConfig.BUCKET_NAME}-dataflow-{secrets.token_hex(3)}-{batch_name.lower()}-{known_args.location}"
    google_cloud_options.staging_location = GCPConfig.STAGING_LOCATION
    google_cloud_options.temp_location = GCPConfig.TEMP_LOCATION
    options.view_as(WorkerOptions).autoscaling_algorithm = "THROUGHPUT_BASED"
    print(f"🚀 Pipeline setup complete for {batch_name}")
    return options
