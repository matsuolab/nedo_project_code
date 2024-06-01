import re

def preprocess_natural_language_text(text):
    removed = []
    
    # HTMLタグの削除
    html_tags = re.findall(r'<.*?>', text)
    removed.extend(html_tags)
    text = re.sub(r'<.*?>', '', text)
    
    # URLの削除
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    removed.extend(urls)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # ダッシュ、中点、カギ括弧の削除
    dash_brackets = re.findall(r'[-―・「」\[\]]', text)
    removed.extend(dash_brackets)
    text = re.sub(r'[-―・「」\[\]]', '', text)
    
    # 全角記号から半角記号への変換
    full_width_chars = re.findall(r'[！-～]', text)
    removed.extend(full_width_chars)
    text = text.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))
    
    # 絵文字の削除
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    emojis = emoji_pattern.findall(text)
    removed.extend(emojis)
    text = emoji_pattern.sub(r'', text)
    
    return text, list(set(removed))



def preprocess_medical_scientific_text(text):
    removed = []
    # 上付き文字の処理
    removed.extend(re.findall(r'([a-zA-Z0-9])\^([a-zA-Z0-9])', text))
    text = re.sub(r'([a-zA-Z0-9])\^([a-zA-Z0-9])', r'\1^\2', text)
    # 下付き文字の処理
    removed.extend(re.findall(r'([a-zA-Z0-9])_([a-zA-Z0-9])', text))
    text = re.sub(r'([a-zA-Z0-9])_([a-zA-Z0-9])', r'\1_\2', text)
    return text, removed


def preprocess_financial_economic_text(text):
    removed = []
    # ハイフンの削除
    removed.extend(re.findall(r'-', text))
    text = re.sub(r'-', ' ', text)
    return text, removed


def preprocess_social_media_text(text):
    removed = []
    # URLの削除
    removed.extend(re.findall(r'https?://\S+|www\.\S+', text))
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 絵文字の削除
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    removed.extend(emoji_pattern.findall(text))
    text = emoji_pattern.sub(r'', text)
    # 特殊な省略形の展開
    removed.extend(re.findall(r'\b(w/)\b', text))
    text = re.sub(r'\b(w/)\b', 'with', text)
    removed.extend(re.findall(r'\b(b/c)\b', text))
    text = re.sub(r'\b(b/c)\b', 'because', text)
    return text, removed


def preprocess_legal_contract_text(text):
    removed = []
    # 箇条書き記号の削除
    removed.extend(re.findall(r'[・-]', text))
    text = re.sub(r'[・-]', '', text)
    # 参照記号の削除
    removed.extend(re.findall(r'(上記|下記|同上)', text))
    text = re.sub(r'(上記|下記|同上)', '', text)
    return text, removed


def preprocess_mathematical_text(text):
    removed = []
    # 桁区切りのカンマの削除
    removed.extend(re.findall(r'(\d),(\d)', text))
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # コメントアウトの記号の削除
    removed.extend(re.findall(r'//.*', text))
    text = re.sub(r'//.*', '', text)
    removed.extend(re.findall(r'/\*.*?\*/', text, flags=re.DOTALL))
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text, removed


def preprocess_latex_math(text):
    removed = []

    # インラインの数式を抽出し、トークンに置換
    inline_math = re.findall(r'\$(.+?)\$', text)
    removed.extend(inline_math)
    text = re.sub(r'\$(.+?)\$', replace_inline_math, text)

    # ディスプレイスタイルの数式を抽出し、トークンに置換
    display_math = re.findall(r'\$\$(.+?)\$\$', text)
    removed.extend(display_math)
    text = re.sub(r'\$\$(.+?)\$\$', replace_display_math, text)

    display_math = re.findall(r'\\\[(.+?)\\\]', text)
    removed.extend(display_math)
    text = re.sub(r'\\\[(.+?)\\\]', replace_display_math, text)

    return text, removed


def preprocess_medical_scientific_text(text):
    # 上付き文字の処理
    text = re.sub(r'([a-zA-Z0-9])\^([a-zA-Z0-9])', r'\1^\2', text)
    # 下付き文字の処理
    text = re.sub(r'([a-zA-Z0-9])_([a-zA-Z0-9])', r'\1_\2', text)
    return text