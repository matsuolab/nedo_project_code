from .parts_filter import filter

noise_ending_list = """
一覧ページ上部へ
ページ上部へ
上部へ
上へ戻る
"""
noise_ending_list = noise_ending_list.split("\n")
noise_ending_list = [x for x in noise_ending_list if len(x) > 0]


def clean_endings(sent: str):
    for noise_ending in noise_ending_list:
        if sent.endswith(noise_ending):
            return sent[:-len(noise_ending)]
    return sent


noise_header_list = """
トップページ>
"""
noise_header_list = noise_header_list.split("\n")
noise_header_list = [x for x in noise_header_list if len(x) > 0]


def clean_headers(text):
    for noise_header in noise_header_list:
        if text.startswith(noise_header):
            return text[len(noise_header):]
    return text


sentence_endings = ['。', '！', '？', '.', '!', '?', "．", "」", "。"]

# 文頭の見出しを消す


def remove_header(txt, header_list, n_check=30):
    for header in header_list:
        if header in txt[:n_check]:
            for delimiter in header_list:
                txt = txt.split(delimiter)[1:]
                txt = delimiter.join(txt)
                break
    return txt


def clean(text):
    text = remove_header(text, header_list=["|", "】", ">", "]",])
    text = clean_endings(text)
    text = clean_headers(text)
    text = dedup_lines(text)

    # 文章全体で名詞が多い場合は無効と判定
    try:
        if not filter(text, threshold=0.7):
            return ""
    except:
        return ""
    for ending in sentence_endings:
        if text.find(ending) > 0:
            return text

    return ""


def dedup_lines(data, check_length=10):

    lines = data.split('\n')
    new_lines = []
    old_line = ""
    for line in lines:
        set_a = set(line[:check_length])
        set_b = set(old_line[:check_length])
        if len(list(set_a-set_b)) < 2:
            continue
        old_line = line
        new_lines.append(line)

    result_text = '\n'.join(new_lines)
    return result_text
