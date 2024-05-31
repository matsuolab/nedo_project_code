from typing import List
import functools
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation


def text_to_paragraph_sentences(text: str) -> List[List[str]]:
    """
    Split a text into paragraphs and sentences.

    Args:
        text (str): The text to split.

    Returns:
        List[List[str]]: The paragraphs and sentences.
    """
    paragraphs = paragraph_split(text)
    return [sentence_split(paragraph) for paragraph in paragraphs]


def paragraph_split(text: str) -> List[str]:
    """
    Split a text into paragraphs.

    Args:
        text (str): The text to split.

    Returns:
        List[str]: The paragraphs.
    """
    return text.split("\n")


split_punc2 = functools.partial(split_punctuation, punctuations=r"。!?")
concat_tail_te = functools.partial(
    concatenate_matching, remove_former_matched=False)
segmenter = make_pipeline(normalize, split_newline,
                          concat_tail_te, split_punc2)


def sentence_split(text: str) -> List[str]:
    """
    Split a text into sentences.

    Args:
        text (str): The text to split.

    Returns:
        List[str]: The sentences.
    """
    return list(segmenter(text))
