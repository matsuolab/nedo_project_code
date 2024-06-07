from fugashi import Tagger
from tqdm import tqdm
import argparse

# 引数設定
def parse_arguments():
    parser = argparse.ArgumentParser() # パーサを作る
    parser.add_argument("--input", type=str, required=True) # インプットファイルの指定
    parser.add_argument("--output", type=str, required=True) # アウトプットファイルの指定
    args = parser.parse_args() # 引数を解析
    print(f"{args = }") # 引数を表示
    return args # 引数を返す

# 改行ごとに形態素解析を行う
def process(input_file: str, output_file: str):
    # 全体のステップ数を取得
    with open(input_file, "r") as reader:
        lines = reader.readlines()  # ファイルを一度に読み込む
        print("全ステップ数：", len(lines))

    # 形態素解析を行い、ファイルに書き込む
    with open(output_file, "w") as writer:
        tagger = Tagger("-Owakati")
        for line in tqdm(lines, desc="Processing"): # 進捗を表示
            wakati = tagger.parse(line)
            custom_wakati = "||||".join(wakati.split()) # 空白で区切られた単語を"||||"で再結合
            writer.write(custom_wakati + "\n")
            
# メイン関数
def main():
    args = parse_arguments() # 引数を取得
    process(args.input, args.output) # 形態素解析を行う
    
# 実行
if __name__ == "__main__":
    main()
