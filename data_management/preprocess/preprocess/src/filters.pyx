import cython
import re
import zlib
import math
import mmh3
import unicodedata
from pathlib import Path
from urllib.parse import urlparse


HIRAGANA_KATAKANA_PAT = re.compile(r"[ぁ-んァ-ン]")
JAPANESE_SEPARATER_PAT = re.compile(r"。|、|．|，|｡|､")


def has_valid_domain(dict_path, exclude_wikipedia=True):
    dict_path = Path(dict_path)
    valid_domains = set(dict_path.read_text().splitlines()) - {''}
    def domain_checker(str url):
        if exclude_wikipedia and url.startswith('https://ja.wikipedia.org'):
            return False
        domain = urlparse(url).hostname
        if domain is None:
            print(url)
            return False
        tld = domain.split(".")[-1]
        return tld in valid_domains
    return domain_checker


def normalize(str text):
    return unicodedata.normalize("NFKC", text)


def is_not_empty(str text):
    return text.strip() != ""


def is_japanese(str text, cython.int lookup_size = 50):
    if not HIRAGANA_KATAKANA_PAT.search(text[:lookup_size]):
        return False
    return True


def gen_NG_word_counter(dict_path, ignore_confused=False):
    dict_path = Path(dict_path)
    dict_NG_word = set(dict_path.read_text().splitlines()) - {''}
    if ignore_confused:
        words_katakana = []
        words_not_katakana = []
        for w in dict_NG_word:
            if re.fullmatch(r"[ァ-ヴー]+", w):
                words_katakana.append(re.escape(w))
            else:
                words_not_katakana.append(re.escape(w))
        katakana_pat = "|".join(words_katakana)
        katakana_pat = rf"(?<![ァ-ヴー])({katakana_pat})(?![ァ-ヴー])"
        pat = "|".join(words_not_katakana) + "|" + katakana_pat
        pat = re.compile(pat)
    else:
        pat = re.compile(r"|".join(dict_NG_word))
    def count_NG_word(text: str):
        factor = pat.findall(text)
        return len(factor)
    return count_NG_word


def average_sentence_length(str text):
    sentences = JAPANESE_SEPARATER_PAT.findall(text)
    num_separater: int = len(sentences)
    return len(text) // (num_separater + 1)


def compression_ratio(str text, cython.double length_factor = 0.0):
    encoded = text.encode('utf-8')
    compressed = zlib.compress(encoded, level=9)
    cdef cython.double encoded_length = len(encoded)
    cdef cython.double compressed_length = len(compressed)
    cdef cython.double ratio = compressed_length / encoded_length
    cdef cython.double length_penalty = (
        length_factor * math.log(encoded_length) if length_factor else 0.0
    )
    score = ratio + length_penalty
    return score
