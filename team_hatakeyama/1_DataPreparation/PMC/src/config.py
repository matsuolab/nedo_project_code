import os


class GCPConfig:
    PROJECT_ID = os.getenv("GCP_PROJECT_ID", "geniac-416410")
    CREDENTIAL_PATH = os.getenv(
        "GCP_CREDENTIAL_PATH", "sec/geniac-416410-5bded920e947.json"
    )
    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "geniac-pmc")
    REGION = os.getenv("GCP_REGION", "us-east1")
    MACHINE_TYPE = os.getenv("GCP_MACHINE_TYPE", "n1-standard-1")
    STAGING_LOCATION = f"gs://{BUCKET_NAME}/binaries"
    TEMP_LOCATION = f"gs://{BUCKET_NAME}/temp"


class HFConfig:
    ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
    REPO_ID = os.getenv("HF_REPO_ID", "hatakeyama-llm-team/PMC")
    MAX_FILE_SIZE = int(49.9 * 1000**3)
