import asyncio
import time
from pipeline_setup import cli_args
from batch_process import run_batch_async


async def main():
    known_args, _ = cli_args()
    for batch in range(known_args.start_batch, known_args.end_batch + 1):
        batch_name = f"PMC{str(batch).zfill(3)}xxxxxx"
        print(f"🔥 Starting processing for {batch_name}")
        start_time = time.time()
        await run_batch_async(batch_name)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"🕒 Batch {batch_name} completed in {execution_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
