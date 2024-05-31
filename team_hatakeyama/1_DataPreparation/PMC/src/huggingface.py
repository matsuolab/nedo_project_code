import asyncio
import logging
import os
from huggingface_hub import HfApi
from config import HFConfig

api = HfApi()


async def upload_to_huggingface(output_path):
    if not output_path:
        logging.error("No output path provided for uploading.")
        return

    if not os.path.isfile(output_path):
        logging.error(f"File {output_path} does not exist on the local file system.")
        return

    try:
        print(f"😳 Uploading {output_path} to Hugging Face.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: api.upload_file(
                token=HFConfig.ACCESS_TOKEN,
                repo_id=HFConfig.REPO_ID,
                path_in_repo=os.path.basename(output_path),
                path_or_fileobj=output_path,
                repo_type="dataset",
            ),
        )
        print("🤗 Upload completed successfully.")
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"🗑️ Deleted local file {output_path}")
    except Exception as e:
        logging.error(f"Failed to upload {output_path} to Hugging Face: {e}")
        raise
