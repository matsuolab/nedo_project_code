from collections import Counter

# 単一文字の繰り返し200----------------------------------------


def repeated_id(text, threshold_ratio=0.3):
    # # 文字の繰り返しをチェックする辞書
    char_count = {}
    # 各文字の繰り返し回数をカウント
    for char in text:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    # ratio%以上繰り返される文字があるかチェック
    repeated_chars = [char for char, count in char_count.items(
    ) if count >= threshold_ratio*len(text)]
    # 繰り返される文字がなければmatrix_tempに追加
    if not repeated_chars:
        return text
    return ""


def is_repetitive_japanese(text, thresholds={
    'line_dup': 0.30,
    'paragraph_dup': 0.30,
    'char_in_line_dup': 0.20,
    'char_in_paragraph_dup': 0.20,
    '2-gram': 0.20,
    '3-gram': 0.18,
    '4-gram': 0.16,
    '5-gram': 0.15,
    '6-gram': 0.14,
    '7-gram': 0.13,
    '8-gram': 0.12,
    '9-gram': 0.11,
        '10-gram': 0.10}):
    if text == "":
        return ""
    # 段落と行に分割
    paragraphs = text.split('\n')
    lines = text.replace('\n', ' ').split('。')

    # 段落と行の重複率を計算
    paragraph_dup_rate = calc_dup_rate(paragraphs)
    line_dup_rate = calc_dup_rate(lines)

    # 文字に含まれる重複の割合を計算
    char_in_paragraph_dup_rate = calc_char_dup_rate(paragraphs, text)
    char_in_line_dup_rate = calc_char_dup_rate(lines, text)

    # n-gramの重複率を計算
    ngram_dup_rates = {}
    for n in range(2, 11):
        ngrams = extract_ngrams(text.replace('\n', ''), n)
        if n < 5:
            # 最頻出のn-gramの出現回数を計算
            ngram_dup_rates[n] = calc_max_freq_rate(ngrams)
        else:
            # 2回以上出現するn-gramの総出現回数を計算
            ngram_dup_rates[n] = calc_total_dup_freq_rate(ngrams)

    # 各指標が閾値を超えているかチェック
    if (line_dup_rate > thresholds['line_dup'] or
        paragraph_dup_rate > thresholds['paragraph_dup'] or
        char_in_line_dup_rate > thresholds['char_in_line_dup'] or
        char_in_paragraph_dup_rate > thresholds['char_in_paragraph_dup'] or
            any(ngram_dup_rates[n] > thresholds[f'{n}-gram'] for n in range(2, 11))):
        # print(text)
        return ""

    return text


def calc_dup_rate(items):
    counter = Counter(items)
    total = len(items)
    dup_count = sum(1 for count in counter.values() if count > 1)
    return dup_count / total if total > 0 else 0


def calc_char_dup_rate(items, text):
    counter = Counter(items)
    total_chars = len(text)
    dup_chars = sum(len(item) * (count - 1)
                    for item, count in counter.items() if count > 1)
    return dup_chars / total_chars if total_chars > 0 else 0


def extract_ngrams(text, n):
    return [text[i:i+n] for i in range(len(text)-n+1)]


def calc_max_freq_rate(ngrams):
    counter = Counter(ngrams)
    total = len(ngrams)
    max_freq = max(counter.values())
    return max_freq / total if total > 0 else 0


def calc_total_dup_freq_rate(ngrams):
    counter = Counter(ngrams)
    total = len(ngrams)
    dup_freq = sum(count - 1 for count in counter.values() if count > 1)
    return dup_freq / total if total > 0 else 0

# text = "これはテストです。これはテストです。これはテストです。\nこれもテストです。これもテストです。"

# print(text)
# print(is_repetitive_japanese(text, thresholds))  # 空の文字列を出力
