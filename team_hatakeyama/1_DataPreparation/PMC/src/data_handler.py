import tarfile
from google.cloud import storage
import asyncio
from config import GCPConfig, HFConfig
import os
import json
import logging
import aiofiles
import shutil


async def download_and_extract_tar(batch_name):
    print("⏬ Downloading tar")
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCPConfig.BUCKET_NAME)
    tar_filename = f"oa_comm_xml.{batch_name}.baseline.2023-12-18.tar.gz"
    tar_path = f"original_files/{tar_filename}"
    destination_path = "xml_files"

    blob = bucket.blob(tar_path)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, blob.download_to_filename, tar_filename)

    print("🗃️  Extracting tar")
    with tarfile.open(tar_filename, "r:gz") as tar:
        await loop.run_in_executor(None, tar.extractall, destination_path)
    os.remove(tar_filename)


async def combine_json_files(batch_name):
    json_dir = f"jsonl_files/{batch_name}/"
    file_counter = 0
    current_file_size = 0
    current_file_path = f"{batch_name}_{file_counter}.jsonl"
    file_stream = await aiofiles.open(current_file_path, "w")

    try:
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        total_json_files = len(json_files)

        for i, filename in enumerate(json_files, start=1):
            filepath = os.path.join(json_dir, filename)
            async with aiofiles.open(filepath, "r") as infile:
                data = json.loads(await infile.read())
                text = json.dumps({"text": data["text"]})
                if (
                    current_file_size + len(text.encode("utf-8"))
                    > HFConfig.MAX_FILE_SIZE
                ):
                    await file_stream.close()
                    file_counter += 1
                    current_file_path = f"{batch_name}_{file_counter}.jsonl"
                    file_stream = await aiofiles.open(current_file_path, "w")
                    current_file_size = 0
                await file_stream.write(text + "\n")
                current_file_size += len(text.encode("utf-8"))

            progress_message = f"🔮 Combining JSONL files: {i}/{total_json_files}"
            print(f"\r{progress_message}", end="", flush=True)

        print("\n🍻 Successfully combined JSONL files into: " + current_file_path)
    except Exception as e:
        logging.error(
            f"💀 Failed to combine JSONL files for batch {batch_name}: {e}",
            exc_info=True,
        )
    finally:
        await file_stream.close()
        shutil.rmtree(json_dir)

    return [f"{batch_name}_{i}.jsonl" for i in range(file_counter + 1)]
