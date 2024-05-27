begin_symbols = [
    "(",
    "[",
    "{",
    "（",
    "［",
    "｛",
    "〈",
    "《",
    "「",
    "『",
    "【",
    "〔",
    "・",
    "、",
    ",",
    "，",
]

end_symbols = [
    ")",
    "]",
    "}",
    "）",
    "］",
    "｝",
    "〉",
    "》",
    "」",
    "』",
    "】",
    "〕",
]


def is_end_with_begin_symbol(sent):
    for s in begin_symbols:
        if sent.endswith(s):
            return True
    return False


def is_start_with_end_symbol(sent):
    for s in end_symbols:
        if sent.startswith(s):
            return True
    return False


def is_sentence_end(sent):
    end_symbols = [".", "。", "!", "！", "?", "？", "ました", "ます",
                   "れる", "する", "すよ", "です", "」", "』", "さい",]
    for s in end_symbols:
        if sent.endswith(s):
            return True
    return False


def remove_multi_headers(new_texts):
    """
    見出し1
    見出し2
    見出し3
    こんにちは｡

    →
    見出し3
    こんにちは｡

    にする
    """
    new_texts = new_texts.split("\n")
    new_texts = remove_dup_lines(new_texts)
    cleaned_texts = []
    cap_line = ""
    for line in new_texts[::-1]:
        if not is_sentence_end(line):
            # print("line",line)
            if cap_line == "":
                cap_line = line
                # print("cap_line",cap_line)

        else:
            if cap_line != "":
                cleaned_texts.append(cap_line)
                cap_line = ""

            cleaned_texts.append(line)
    cleaned_texts.append(cap_line)
    cleaned_texts = cleaned_texts[::-1]
    cleaned_texts = [i.strip() for i in cleaned_texts]
    cleaned_texts = "\n".join(cleaned_texts)
    return cleaned_texts


def remove_dup_lines(lines, dup_n_threshold=100):
    new_lines = []
    for line in lines:
        # line=line[0].strip()

        # 行の重複を避ける
        if line in new_lines[-dup_n_threshold:]:
            continue
        new_lines.append(line)

    return new_lines
