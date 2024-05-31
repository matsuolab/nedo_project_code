import logging
from huggingface import upload_to_huggingface
from data_handler import download_and_extract_tar, combine_json_files
from xml_processing import process_batch


async def run_batch_async(batch_name):
    await download_and_extract_tar(batch_name)
    processed_files = await process_batch(batch_name)

    if processed_files and len(processed_files) > 0:
        combined_jsonl_paths = await combine_json_files(batch_name)
        if combined_jsonl_paths and isinstance(combined_jsonl_paths, list):
            for path in combined_jsonl_paths:
                await upload_to_huggingface(path)
        else:
            logging.error("No JSONL files were generated to upload.")
    else:
        logging.error("Failed to process files or no files were processed.")
