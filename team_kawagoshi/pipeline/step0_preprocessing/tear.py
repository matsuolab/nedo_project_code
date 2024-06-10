import pyarrow.parquet as pq

def extract_first_100_rows(input_file: str, output_file: str):
    table = pq.read_table(input_file)
    first_100_rows = table.slice(0, 1000)
    pq.write_table(first_100_rows, output_file)

# 使用例
input_file = "../dataset/train_0.parquet"
output_file = "../small_dataset/train_0_head.parquet"
extract_first_100_rows(input_file, output_file)