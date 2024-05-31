import os
import json
import pandas as pd
from text_extraction import generate_record
import logging
import asyncio
import aiofiles
import shutil


async def process_xml_file(filepath):
    try:
        loop = asyncio.get_running_loop()
        with open(filepath, "r") as file:
            xml_string = await loop.run_in_executor(None, file.read)

        if not xml_string:
            logging.warning(f"🦴 File is empty: {filepath}")
            return None

        # Ensure the generate_record function can handle empty strings gracefully
        record = (
            await generate_record(xml_string)
            if asyncio.iscoroutinefunction(generate_record)
            else generate_record(xml_string)
        )
        if not record:
            logging.warning(f"🦴 No content extracted from XML: {filepath}")
            return None
        return {"text": record, "filepath": filepath}
    except FileNotFoundError:
        logging.error(f"🚫 File not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"❌ Failed to process file {filepath}: {e}", exc_info=True)
        return None


async def write_to_json(record, batch_name):
    if record and record["text"]:
        filepath = record["filepath"]
        file_name = os.path.basename(filepath).replace(".xml", ".json")
        output_path = os.path.join("jsonl_files", batch_name, file_name)

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            async with aiofiles.open(output_path, "w") as json_file:
                await json_file.write(json.dumps(record))
            return output_path
        except Exception as e:
            logging.error(
                f"❌ Failed to write record to JSON: {output_path}: {e}", exc_info=True
            )
            return None
    else:
        return None


async def process_batch(batch_name):
    csv_path = f"target/{batch_name}.csv"
    try:
        df = pd.read_csv(csv_path, header=None, skiprows=1)
        xml_filenames = df.iloc[:, 0].dropna().tolist()
        total_files = len(xml_filenames)
        print(f"📒 Number of valid XML filenames extracted: {total_files}")

        if not xml_filenames:
            logging.error("🚧 No XML files found for processing.")
            return []

        json_file_paths = []
        for i, filename in enumerate(xml_filenames, start=1):
            print(f"\r🏎️  Processing files: {i}/{total_files}", end="")
            filepath = os.path.join("xml_files", filename)
            record = await process_xml_file(filepath)
            if record:
                json_path = await write_to_json(record, batch_name)
                if json_path:
                    json_file_paths.append(json_path)

        return json_file_paths
    except FileNotFoundError:
        logging.error(f"🚧 CSV file not found: {csv_path}")
        return []
    except Exception as e:
        logging.error(f"💀 Failed to process batch {batch_name}: {e}", exc_info=True)
        return []
    finally:
        shutil.rmtree(f"xml_files/{batch_name}")
