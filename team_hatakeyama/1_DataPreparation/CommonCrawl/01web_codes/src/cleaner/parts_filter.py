

import MeCab
from collections import Counter

# テキスト
tagger = MeCab.Tagger()


def parts_count(text, return_word_count=False):
    # テキストを解析
    parsed = tagger.parse(text)

    # 品詞をカウントするためのCounterオブジェクト
    pos_counter = Counter()
    word_counter = Counter()

    # 解析結果を行ごとに処理
    all_counts = 0
    for line in parsed.split('\n'):
        # EOSまたは空行の場合はスキップ
        if line == 'EOS' or line == '':
            continue
        # タブで分割し、形態素情報を取得
        pos_info = line.split('\t')
        # print(pos_info)
        pos = pos_info[1]
        pos = pos.split(",")[0]

        if return_word_count:
            word = pos_info[0]
            word_counter[(word, pos)] += 1

        # 品詞をカウント
        pos_counter[pos] += 1
        all_counts += 1

    if return_word_count:
        return pos_counter, all_counts, word_counter
    else:
        return pos_counter, all_counts


def filter(text, threshold=0.9, min_length=10):
    """
    名詞の羅列の文章は無効と判定する
    """
    if text is None:
        return None
    if text == "":
        return None
    pos_counter, all_counts = parts_count(text)
    # print(pos_counter, all_counts)
    # print(pos_counter)
    meishi_and_symbol_counts = pos_counter['名詞'] + \
        pos_counter['記号']+pos_counter['補助記号']+pos_counter['接頭詞']

    if all_counts == 0:
        return None
    ratio = meishi_and_symbol_counts/all_counts
    # print(ratio, pos_counter)
    if ratio > threshold and len(text) > min_length:
        return None
    else:
        return text


def n_gram(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]

# +(n-gramによって重複の有無を調べる)


def filter2(text, threshold=0.9, min_length=10,  ngram_threshold_2gram=0.20, ngram_threshold_3gram=0.20, ngram_threshold_4gram=0.20):
    if text is None:
        return None

    pos_counter, words, word_counter = parts_count(
        text, return_word_count=True)
    meishi_and_symbol_counts = pos_counter['名詞'] + \
        pos_counter['記号'] + pos_counter['補助記号']
    ratio = meishi_and_symbol_counts / len(words)

    if ratio > threshold and len(text) > min_length:
        return None

    # 2-gramの処理
    ngram_counts_2gram = Counter(n_gram(words, 2))
    total_2grams = sum(ngram_counts_2gram.values())
    most_common_2gram_count = ngram_counts_2gram.most_common(
        1)[0][1] if ngram_counts_2gram else 0
    if total_2grams > 0 and most_common_2gram_count / total_2grams > ngram_threshold_2gram:
        return None

    # 3-gramの処理
    ngram_counts_3gram = Counter(n_gram(words, 3))
    total_3grams = sum(ngram_counts_3gram.values())
    repeated_3grams = sum(
        count for count in ngram_counts_3gram.values() if count > 1)
    if total_3grams > 0 and repeated_3grams / total_3grams > ngram_threshold_3gram:
        return None

    # 4-gramの処理
    ngram_counts_4gram = Counter(n_gram(words, 4))
    total_4grams = sum(ngram_counts_4gram.values())
    repeated_4grams = sum(
        count for count in ngram_counts_4gram.values() if count > 1)
    if total_4grams > 0 and repeated_4grams / total_4grams > ngram_threshold_4gram:
        return None

    return text
