import argparse
import os
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tokenizer_and_model_dir", type=str, required=True,)
    args = parser.parse_args()
    print(f"{args = }")
    return args


def load_tokenizer_and_model(input_tokenizer_and_model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_and_model_dir)
    model = AutoModelForCausalLM.from_pretrained(input_tokenizer_and_model_dir, device_map="auto",
                                                torch_dtype=torch.bfloat16)
    return tokenizer, model


def test_tokenizer_and_model(tokenizer, model, prompt_text: str) -> str:
    encoded_prompt_text = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        encoded_generation_text = model.generate(encoded_prompt_text, max_new_tokens=100, temperature=0.1, do_sample=True)[0]
    decoded_generation_text = tokenizer.decode(encoded_generation_text)
    return decoded_generation_text


def main() -> None:
    args = parse_arguments()

    test_prompt1 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:
前提と仮説の関係をentailment、contradiction、neutralの中から回答してください。それ以外には何も含めないことを厳守してください。

制約：
- 前提から仮説が、時間関係上導出可能である場合はentailmentと出力
- 前提と仮説が両立しえない場合はcontradictionと出力
- そのいずれでもない場合はneutralと出力

### 入力:
前提：ボブは選挙に撤回を表明している。
仮説：ボブは現在選挙に撤回を表明している。

### 応答:
"""

    test_prompt2 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:
質問に対する答えを出力してください。答えが複数の場合、コンマ（,）で繋げてください。

### 入力:
質問：日本の初代首相は？

### 応答:
"""

    test_prompt3 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示:
与えられたテキストから同一の対象を指し示すフレーズを全て抽出してください。回答は以下のような形式で答えてください。
フレーズ1 フレーズ2 フレーズ3
フレーズ4 フレーズ5

### 入力:
就業規則とは、企業において使用者が労働基準法等に基づき、当該企業における労働条件等に関する具体的細目について定めた規則集のことをいう。

### 応答:
"""
    test_prompt4 = """Q:
日本の総理大臣は? A:"""

    test_prompt5 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\nヴァージン・オーストラリア航空はいつから運航を開始したのですか？\n\n### 入力:\nヴァージン・オーストラリア航空（Virgin Australia Airlines Pty Ltd）はオーストラリアを拠点とするヴァージン・ブランドを冠する最大の船団規模を持つ航空会社です。2000年8月31日に、ヴァージン・ブルー空港として、2機の航空機、1つの空路を運行してサービスを開始しました。2001年9月のアンセット・オーストラリア空港の崩壊後、オーストラリアの国内市場で急速に地位を確立しました。その後はブリスベン、メルボルン、シドニーをハブとして、オーストラリア国内の32都市に直接乗り入れるまでに成長しました。\n\n### 応答:\n"""
    test_prompt6 = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n### 指示:\nロッククライミングで使用されるハーネスの種類をリストアップしてください。\n\n### 入力:\nクライミングの種類によって、ハーネスに求められる機能は異なります。スポーツクライミングでは、最小限のハーネスを使用し、ギアループを縫い付けているものもあります。アルパインクライマーは、軽量なハーネスを選ぶことが多く、レッグループが取り外し可能なものもあります。ビッグウォールクライマーは、パッド入りのウエストベルトとレッグループを好みます。また、骨盤が狭く、標準的なハーネスを安全に支えることができない子供用のフルボディハーネスもあります。倒立しても落下しないように、子供用として製造されたものや、ウェビングで作られたものがあります。倒立の可能性があるときや、重いザックを背負うときは、全身ハーネスを使用する人もいます。また、シットハーネスと組み合わせて使用するチェストハーネスもあります。UIAAの試験結果によると、チェストハーネスはシットハーネスよりも首への衝撃が少なく、フルボディハーネスと同じ利点があります[6]。\n\nこれらのハーネスとは別に、ケイビングハーネスやキャニオニングハーネスがあり、それぞれ異なる用途で使用されています。例えば、ケイビング用ハーネスは、防水でパッドなしの丈夫な素材でできており、2つのアタッチメントポイントがあります。この取り付け部分からマイヨンを離すと、ハーネスが素早く緩みます。\n\nキャニオニング用ハーネスは、クライミング用ハーネスに似ていて、パッドはありませんが、シートプロテクターが付いていて、懸垂下降をより快適に行うことができます。これらのハーネスは通常、ダイニーマの1つのアタッチメントポイントを持っています。### 応答:\n"""
    test_prompt7 = """Q: 日本の首都は? A:"""
    # Loads and tests the local tokenizer and the local model.
    local_tokenizer, local_model = load_tokenizer_and_model(args.input_tokenizer_and_model_dir)
    local_decoded_generation_text = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt1)
    local_decoded_generation_text2 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt2)
    local_decoded_generation_text3 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt3)
    local_decoded_generation_text4 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt4)
    local_decoded_generation_text5 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt5)
    local_decoded_generation_text6 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt6)
    local_decoded_generation_text7 = test_tokenizer_and_model(local_tokenizer, local_model, test_prompt7)
    # Checks the generated text briefly.
    print()
    print(f"{test_prompt1 = }")
    print(f"{local_decoded_generation_text = }")
    print()
    print()
    print(f"{test_prompt2 = }")
    print(f"{local_decoded_generation_text2 = }")
    print()
    print(f"{test_prompt3 = }")
    print(f"{local_decoded_generation_text3 = }")
    print()
    print(f"{test_prompt4 = }")
    print(f"{local_decoded_generation_text4 = }")
    print()
    print(f"{test_prompt5 = }")
    print(f"{local_decoded_generation_text5 = }")
    print()
    print(f"{test_prompt6 = }")
    print(f"{local_decoded_generation_text6 = }")
    print()
    print(f"{test_prompt7 = }")
    print(f"{local_decoded_generation_text7 = }")
    # Loads and tests the remote tokenizer and the remote model.
    #huggingface_username = HfApi().whoami()["name"]
    #remote_tokenizer, remote_model = load_tokenizer_and_model(os.path.join(huggingface_username, args.output_model_name))
    #remote_decoded_generation_text = test_tokenizer_and_model(remote_tokenizer, remote_model, args.test_prompt_text)

    # Checks the generated text briefly.
    #print()
    #print(f"{args.test_prompt_text = }")
    #print(f"{remote_decoded_generation_text = }")
    #print()
    #if len(remote_decoded_generation_text) <= len(args.test_prompt_text):
    #    print("Error: The generated text should not be shorter than the prompt text."
    #          " Something went wrong, so please check either the remote tokenizer or the remote model.")
    #    return


if __name__ == "__main__":
    main()
