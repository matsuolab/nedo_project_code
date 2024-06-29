import asyncio
from dotenv import dotenv_values
import argparse
from openai import AsyncOpenAI
from string import Template
from datetime import datetime
import pandas as pd
import json
import time
import random
import io

# 引数を解析する
parser = argparse.ArgumentParser()
parser.add_argument(
    "--api_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--output_path",
    type=str,
    default="test/mixtral/data/output",
)
parser.add_argument(
    "--prompt_path",
    type=str,
    default="test/prompt/system.md",
)
parser.add_argument(
    "--shot_sample_path",
    type=str,
    default="test/shot_sample/grammaticality.json",
)
parser.add_argument(
    "--temperature",
    type=int,
    default=1,
)
parser.add_argument(
    "--max_tokens",
    type=int,
    default=10000,
)
parser.add_argument(
    "--top_p",
    type=int,
    default=0.9,
)
parser.add_argument(
    "--frequency_penalty",
    type=int,
    default=1.3,
)
parser.add_argument(
    "--presence_penalty",
    type=int,
    default=1.3,
)
parser.add_argument(
    "--n_loop",
    type=int,
    default=1,
)
args = parser.parse_args()

# 必要な情報を引数や入力から取得する
api_path = (
    args.api_path
    if args.api_path is not None
    else input("APIキーの書かれた.envファイルへのパスを入力してください: ")
)
output_path = (
    args.output_path
    if args.output_path is not None
    else input("LLMの出力結果を格納するフォルダへのパスを入力してください: ")
)
prompt_path = (
    args.prompt_path
    if args.prompt_path is not None
    else input("プロンプトへのパスを入力してください: ")
)
shot_sample_path = (
    args.shot_sample_path
    if args.shot_sample_path is not None
    else input("ショット例へのパスを入力してください: ")
)
temperature = (
    args.temperature
    if args.temperature is not None
    else input("温度パラメタの値を入力してください: ")
)
max_tokens = (
    args.max_tokens
    if args.max_tokens is not None
    else input("最大入出力トークン数を入力してください: ")
)
top_p = (
    args.top_p
    if args.top_p is not None
    else input("top_pパラメタの値を入力してください: ")
)
frequency_penalty = (
    args.frequency_penalty
    if args.frequency_penalty is not None
    else input("frequency_penaltyパラメタの値を入力してください: ")
)
presence_penalty = (
    args.presence_penalty
    if args.presence_penalty is not None
    else input("presence_penaltyパラメタの値を入力してください: ")
)
n_loop = (
    args.n_loop
    if args.n_loop is not None
    else input("ループ数を入力してください（ループ1回あたり10件データを作成します）: ")
)

# .envファイルから設定を読み込む
config = dotenv_values(api_path)
DEEPINFRA_API_KEY = config["DEEPINFRA_API_KEY"]

# OpenAIクライアントを作成
openai = AsyncOpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

# プロンプトとショット例をファイルから読み込む
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt_template = f.read()
with open(shot_sample_path, "r", encoding="utf-8") as f:
    fewshot_examples = json.load(f)  # f.read()


def generate_fewshot_examples(fewshot_examples, sample_num=10):

    # データをランダムに選択
    selected_data = random.sample(fewshot_examples, sample_num)

    # データを並び替え
    random.shuffle(selected_data)

    # idキーの値を振り直し
    for i, item in enumerate(selected_data, 1):
        item["id"] = i

    # JSONL形式の文字列に変換する
    jsonl_str = "\n".join(
        json.dumps(item, ensure_ascii=False) for item in selected_data
    )

    return jsonl_str


# 非同期でモデルを実行する関数
async def run_model_async(prompt_template, temperature, max_tokens):
    chat_completion = await openai.chat.completions.create(
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        messages=[
            {
                "role": "system",
                "content": "あなたは日本語の学習データセットを生成する専門家です。",
            },
            {"role": "user", "content": prompt_template},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    output = chat_completion.choices[0].message.content
    return output


# メインの非同期関数
async def main():
    df_list = []
    formatted_prompt_template = Template(prompt_template).substitute(
        FewShotExamples=generate_fewshot_examples(
            fewshot_examples,
            sample_num=10,
        )
        # FewShotExamples=fewshot_examples
    )
    print(formatted_prompt_template)
    for i in range(n_loop):
        for retry_i in range(3):
            try:
                output = await run_model_async(
                    formatted_prompt_template, temperature, max_tokens
                )
                print(output)
                df_output = pd.read_json(
                    io.StringIO(output),
                    orient="records",
                    lines=True,
                )
                df_list.append(df_output)
                print("Success parsing JSON")
                break
            except Exception as e:
                print(e)
                print(f"Error parsing JSON: {retry_i}")
                continue
    df_final = pd.concat(df_list)
    df_final.head()
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename_xlsx = f"/generated_data_{timestamp}.xlsx"
    df_final.to_excel(
        output_path + filename_xlsx,
        index=False,
    )
    filename_jsonl = f"/generated_data_{timestamp}.jsonl"
    df_final.to_json(
        output_path + filename_jsonl,
        index=False,
        orient="records",
        force_ascii=False,
        lines=True,
    )
    print(f"File saved as: {filename_xlsx}")


# asyncioを使ってメイン関数を実行
if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()

    print(f"The task is completed in {round(end_time - start_time, 2)} sec.")
