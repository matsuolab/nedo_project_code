from .splitter import text_to_paragraph_sentences
from .text_normalizer import normalize
from . import text_checker
from . import rule_based_line_checker
from . import parts_filter
from .line_end_cleaner import clean_line_endings
from .hojichar_filter import hoji_filter, prob_hoji_filter, prob_filter
from . import rule_based_text_checker
from .line_dedup import remove_multi_headers
from . import repeated_phrase
try:
    from .TextClassifier import TextClassifier
    classifier = TextClassifier()
except:
    print("error loading TextClassifier. install fasttext to use it")


def text_to_cleaned_paragraphs(text):
    text = normalize(text)  # 正規化
    # text = text_checker.check(text)  # ひらがなを含まないテキストは除外

    # 繰り返し表現を除外 by Naito
    text = repeated_phrase.repeated_id(text)
    text = repeated_phrase.is_repetitive_japanese(
        text)  # n-gramの計算(計算量が多そうな場合、削る)

    # パラグラフと文章に分割
    paragraphs = text_to_paragraph_sentences(text)

    new_paragraphs = []
    for paragraph in paragraphs:
        new_lines = []
        old_line = ""
        for line in (paragraph):
            # ルールベースの行チェック
            new_line = rule_based_line_checker.clean(line)

            # 名詞だらけのlineを除外
            try:
                # new_line = parts_filter.filter(new_line)
                new_line = parts_filter.filter2(new_line)  # n-gramによる重複の除外
            except:
                pass
            if new_line:
                if new_line == old_line:
                    continue
                old_line = new_line
                # ppl=perp_checker(new_line)
                # print(ppl,new_line)
                new_lines.append(new_line)

        if new_lines:
            new_paragraphs.append(new_lines)

    # 文末が｡などでおわらないパラグラフ中の文章を削除
    # clean_line_endings(new_paragraphs)

    # パラグラフにまとめる
    cleaned_paragraphs = []
    old_lines = ""
    for paragraph in new_paragraphs:
        lines = "".join(paragraph)
        if lines == old_lines:
            continue
        cleaned_paragraphs.append(lines)
        old_lines = lines

    return cleaned_paragraphs


def clean_text(text, hoji=True):
    if hoji:
        # text = prob_filter(text)
        text = hoji_filter(text)
        text = prob_hoji_filter(text)

    paragraphs = text_to_cleaned_paragraphs(text)
    # print("aa", original_text)
    text = "\n".join(paragraphs)

    text = remove_multi_headers(text)
    text = rule_based_text_checker.clean(text)
    text = text.strip()
    return text


def ml_clean_text(text):
    text = prob_hoji_filter(text)
    text = hoji_filter(text)
    text = classifier.clean(text)
    if text != "":
        text = clean_text(text, hoji=False)
    return text
