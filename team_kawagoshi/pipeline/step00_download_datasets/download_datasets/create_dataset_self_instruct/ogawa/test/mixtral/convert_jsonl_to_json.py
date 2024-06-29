import json
import multiprocessing
import polars as pl
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--output_path",
    type=str,
    default="test/mixtral/data/output",
)
args = parser.parse_args()
input_path = (
    args.input_path
    if args.input_path is not None
    else input("jsonlファイルへのパスを入力してください: ")
)
output_path = (
    args.output_path
    if args.output_path is not None
    else input("出力結果を格納するフォルダへのパスを入力してください: ")
)

instruction_text = """前提と仮説の関係をentailment、contradiction、neutralのラベルの中から選んでください。回答するラベル以外には**絶対に**何も含めないでください。

### ラベル選択基準と回答方法 ###

- 前提から仮説が真であると推論できる場合は、entailmentを選択し、「entailment」とだけ答える
- 前提から仮説が偽であると推論できる場合は、contradictionを選択し、「contradiction」とだけ答える
- 前提から**仮説の真偽が推論できない**場合は、neutralを選択し、「neutral」とだけ答える"""


def process_line(line):
    # JSON行を辞書に変換
    data = json.loads(line)
    # entailment_instanceからデータを抽出し、新しい形式に変換
    processed_data = [
        {
            "input": f"前提: {instance['premise']}\n仮説: {instance['hypothesis']}",
            "output": instance["label"],
        }
        for instance in data["entailment_instance"]
    ]
    return processed_data


def convert_jsonl_to_json(input_path, instruction_text, output_path):
    # multiprocessingを使用してjsonlファイルの各行を並列に処理
    with open(input_path, "r", encoding="utf-8") as file:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(process_line, file)

    # 結果を1つのリストに統合
    flat_results = [item for sublist in results for item in sublist]

    # polarsデータフレームを作成
    df = pl.DataFrame(flat_results)

    # instruction列とcategory列を追加
    df = df.with_columns(
        [
            pl.lit(instruction_text).alias("instruction"),
            pl.lit("recognizing_textual_entailment").alias("category"),
            pl.arange(1, df.height + 1, eager=True).alias("id"),
        ]
    )

    df = df.select(
        [
            "instruction",
            "input",
            "output",
            "category",
            "id",
        ]
    )

    print(df)

    # JSON形式で出力
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"/RTE_generated_data_{timestamp}.json"
    return df.write_json(
        output_path + filename,
        row_oriented=True,
    )


if __name__ == "__main__":
    convert_jsonl_to_json(input_path, instruction_text, output_path)
