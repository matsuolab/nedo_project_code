import spacy
import ginza
import jaconv
from collections import Counter
import re
import argparse
import multiprocessing
import polars as pl
from datetime import datetime

# nlp = spacy.load("ja_ginza")
nlp = spacy.load("ja_ginza_bert_large")


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
parser.add_argument(
    "--test",
    type=bool,
    default=False,
)
args = parser.parse_args()
input_path = (
    args.input_path
    if args.input_path is not None
    else input("txtファイルへのパスを入力してください: ")
)
output_path = (
    args.output_path
    if args.output_path is not None
    else input("出力結果を格納するフォルダへのパスを入力してください: ")
)
test = args.test if args.test is not None else input("テストモードにしますか？: ")


"""
Named entity recognition
"""


def parse_ner(doc, unique=False):
    list_ents = []
    for entity in doc.ents:
        if not entity.label_.endswith("_Other"):
            list_ents.append(entity.text)
            # list_ents.append(f"{entity.text}（{entity.label_}）")

    if unique is True:

        list_ents_count = Counter(list_ents)

        # 登場頻度が多い順に並び替え
        sorted_items = sorted(
            list_ents_count.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # キーのみを抽出してリストにする
        list_ents_sorted = [item[0] for item in sorted_items]

        return "\n".join(list_ents_sorted)
    else:
        return "\n".join(list_ents)


"""
Reading prediction
"""


def parse_rp(doc, katakana=False):
    list_rp = []
    for token in doc:
        if token.pos_ == "PUNCT":
            list_rp.append(token.orth_)
        else:
            if katakana is True:
                list_rp.append(
                    str(
                        ginza.reading_form(
                            token,
                            False,
                        )
                    )
                )
            else:
                list_rp.append(
                    jaconv.kata2hira(
                        str(
                            ginza.reading_form(
                                token,
                                False,
                            )
                        )
                    )
                )

    return "".join(list_rp)


"""
Dependency parsing
"""


def parse_dep(doc, representation="arrow"):
    list_dep = []
    for sent in doc.sents:
        for span in ginza.bunsetu_spans(sent):
            for token in span.lefts:
                # 句読点で終わる場合、それらを除いた文字列を取得
                bunsetu_token = re.sub(
                    r"[、，。．,.?!？！]+$",
                    "",
                    str(ginza.bunsetu_span(token)),
                )
                bunsetu_span = re.sub(r"[、，。．,.?!？！]+$", "", str(span))
                list_dep.append([bunsetu_token, bunsetu_span])
    if representation == "arrow":
        # arrowがTrueの場合、要素を「 → 」で連結
        list_dep = [f"{dep[0]} → {dep[1]}" for dep in list_dep]
    elif representation == "table":
        # arrowがFalseの場合、要素を「| <文字列1つ目> | <文字列2つ目> |」でフォーマット
        list_dep = [f"| {dep[0]} | {dep[1]} |" for dep in list_dep]
        list_dep.insert(0, "| 係り元 | 係り先 |")
        list_dep.insert(1, "|-------|-------|")
    else:
        raise ValueError(
            f"The representation you specified `{representation}` is not allowed."
        )
    return "\n".join(list_dep)


def parse_dep_token(doc, representation="arrow"):
    list_dep = []
    # 単語間の係り受け解析
    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "PUNCT":
                # 句読点で終わる場合、それらを除いた文字列を取得
                list_dep.append([token.head.text, token.text])
    if representation == "arrow":
        # arrowがTrueの場合、要素を「 → 」で連結
        list_dep = [f"{dep[0]} → {dep[1]}" for dep in list_dep]
    elif representation == "table":
        # arrowがFalseの場合、要素を「| <文字列1つ目> | <文字列2つ目> |」でフォーマット
        list_dep = [f"| {dep[0]} | {dep[1]} |" for dep in list_dep]
        list_dep.insert(0, "| 係り元 | 係り先 |")
        list_dep.insert(1, "|-------|-------|")
    else:
        raise ValueError(
            f"The representation you specified `{representation}` is not allowed."
        )

    return "\n".join(list_dep)


"""
Predicate-argument structure analysis
"""


def join_func(elements):
    return "".join(
        [element if isinstance(element, str) else str(element) for element in elements]
    )


def parse_pas(doc):
    result_dict = {}
    for sentence in doc.sents:
        for span in ginza.bunsetu_spans(sentence):
            if span.label_ in {"ADJP", "VP"}:
                for token in span.lefts:
                    dependent_token = ginza.bunsetu_span(token)
                    for word in dependent_token:
                        main_phrase = ginza.phrase(word, join_func=join_func)
                        for head_token in ginza.bunsetu_head_tokens(span):
                            if word.dep_ == "case":
                                case_reading = "".join(word.morph.get("Reading"))
                                if head_token.lemma_ not in result_dict:
                                    result_dict[head_token.lemma_] = []
                                result_dict[head_token.lemma_].append(
                                    f"{case_reading}: {main_phrase}"
                                )
    result_list = []
    for lemma, cases in result_dict.items():
        cases_str = ", ".join(cases)
        result_list.append(f"{lemma} → {cases_str}")
    return "\n".join(result_list)


"""
Coreference resolution
"""

# TBA

inst_ner = "次の文章から、固有名詞をすべて抜き出し、登場順に一覧化してください。"
inst_ner_unique = (
    "次の文章から、固有名詞をすべて抜き出し、登場回数が多い順に一覧化してください。"
)
inst_rp = "次の文章の読み方をひらがなで書いてください。"
inst_rp_katakana = "次の文章の読み方をカタカナで書いてください。"
inst_dp = "次の文章での文節同士の依存関係を説明してください。"
inst_dp_token = "次の文章での単語同士の依存関係を説明してください。"
inst_dp_table = "次の文章での文節同士の依存関係を説明してください。係り元と係り先を表形式で説明してください。"
inst_dp_token_table = "次の文章での単語同士の依存関係を説明してください。係り元と係り先を表形式で説明してください。"
inst_pas = "次の文章における述語項構造関係を説明してください。"
inst_cr = "次の文章において、同一事物を参照している単語を列挙してください。"

inst_common = "回答以外の文は**絶対に**書かないでください。"

instructions = [
    inst_ner,
    inst_ner_unique,
    inst_rp,
    inst_rp_katakana,
    inst_dp,
    inst_dp_token,
    inst_dp_table,
    inst_dp_token_table,
    inst_pas,
    # inst_cr,
]

final_instructions = [x + inst_common for x in instructions]


def txt_to_json(input_path, output_path, test):
    final_df = None

    with open(input_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            line = line.rstrip()
            if test is True and i >= 10:  # 最初の10行だけ処理
                break
            doc = nlp(line)

            df = pl.DataFrame(
                {
                    "instruction": final_instructions,
                    "input": [line] * len(final_instructions),
                    "output": [
                        parse_ner(doc),
                        parse_ner(doc, unique=True),
                        parse_rp(doc),
                        parse_rp(doc, katakana=True),
                        parse_dep(doc),
                        parse_dep_token(doc),
                        parse_dep(doc, representation="table"),
                        parse_dep_token(doc, representation="table"),
                        parse_pas(doc),
                    ],
                }
            )
            # 作成したdfをfinal_dfに連結
            if final_df is None:
                final_df = df
            else:
                final_df = final_df.vstack(df)

    final_df = final_df.with_columns(
        [
            pl.arange(1, final_df.height + 1, eager=True).alias("id"),
        ]
    )

    print(final_df)

    # JSON形式で出力
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    filename_xlsx = f"/FA_generated_data_{timestamp}.xlsx"
    final_df.write_excel(
        workbook=output_path + filename_xlsx,
    )

    filename = f"/FA_generated_data_{timestamp}.json"
    final_df.write_json(
        output_path + filename,
        row_oriented=True,
    )


if __name__ == "__main__":
    txt_to_json(input_path, output_path, test)
