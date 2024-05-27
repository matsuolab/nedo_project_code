
sentence_endings = ['。', '！', '？', '.', '!', '?', "．", "」"]


def clean_line_endings(paragraphs):
    """文末記号以外の文字を削除する"""
    for paragraph in paragraphs:
        if len(paragraph) < 2:
            continue
        for line in paragraph:
            if line[-1] not in sentence_endings:
                paragraph.remove(line)
                # print(line)
