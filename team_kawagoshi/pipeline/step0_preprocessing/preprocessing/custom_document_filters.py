from hojichar import document_filters, Document
from fugashi import Tagger
from hojichar.core.filter_interface import Filter
import unicodedata
from gensim.models.fasttext import FastText
import numpy as np

from os import PathLike
from typing import Any, Union, Optional
import re

tagger = Tagger('-Owakati')

class DiscardAdultContentJa(document_filters.NgWordsFilterJa):
    """
    TokenFilter の実装例です.
    日本語の成人向けコンテンツを閾値に応じて排除します.
    """

    def __init__(
        self,
        dict_path: Union[str, PathLike] = document_filters.BASE_PATH / "dict/adult_keywords_ja.txt",
        threshold: float = 0.1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)
        self.threshold = threshold

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        adult_content_count = len(matches)
        total_words_count = len(tagger.parse(doc.text).split()) # Owakatiで分かち書きして単語数を数える

        if total_words_count > 0 and adult_content_count / total_words_count > self.threshold:
            doc.is_rejected = True # adult keywordsの割合が閾値を超えたらreject

        return doc

class DiscardAds(document_filters.NgWordsFilterJa):
    """
    TokenFilter の実装例です.
    日本語の成人向けコンテンツを閾値に応じて排除します.
    """

    def __init__(
        self,
        dict_path: Union[str, PathLike] = document_filters.BASE_PATH / "dict/advertisement_keywords_ja.txt",
        threshold: float = 0.1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)
        self.threshold = threshold

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        adult_content_count = len(matches)
        total_words_count = len(tagger.parse(doc.text).split()) # Owakatiで分かち書きして単語数を数える

        if total_words_count > 0 and adult_content_count / total_words_count > self.threshold:
            doc.is_rejected = True # adult keywordsの割合が閾値を超えたらreject

        return doc
    
class DiscardDiscriminationContentJa(document_filters.NgWordsFilterJa):
    """
    TokenFilter の実装例です.
    日本語の成人向けコンテンツを閾値に応じて排除します.
    """

    def __init__(
        self,
        dict_path: Union[str, PathLike] = document_filters.BASE_PATH / "dict/discrimination_keywords_ja.txt",
        threshold: float = 0.1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)
        self.threshold = threshold

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        adult_content_count = len(matches)
        total_words_count = len(tagger.parse(doc.text).split()) # Owakatiで分かち書きして単語数を数える

        if total_words_count > 0 and adult_content_count / total_words_count > self.threshold:
            doc.is_rejected = True # adult keywordsの割合が閾値を超えたらreject

        return doc

class DiscardViolenceContentJa(document_filters.NgWordsFilterJa):
    """
    TokenFilter の実装例です.
    日本語の成人向けコンテンツを閾値に応じて排除します.
    """

    def __init__(
        self,
        dict_path: Union[str, PathLike] = document_filters.BASE_PATH / "dict/violence_keywords_ja.txt",
        threshold: float = 0.1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)
        self.threshold = threshold

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        adult_content_count = len(matches)
        total_words_count = len(tagger.parse(doc.text).split()) # Owakatiで分かち書きして単語数を数える

        if total_words_count > 0 and adult_content_count / total_words_count > self.threshold:
            doc.is_rejected = True # adult keywordsの割合が閾値を超えたらreject

        return doc

class DiscardWithCharacterRatio(Filter):
    """
    TokenFilter の実装例です.
    文字種比率によって文書を排除します.
    """

    def __init__(self, 
                 sentence_length_threshold: Optional[int] = 20, #短い方を削除
                 comma_ratio_threshold: Optional[float] = 0.1, #カンマが多い方を削除
                 hiragana_ratio_threshold: Optional[float] = 0.2, #ひらがなが少ない方を削除
                 katakana_ratio_threshold: Optional[float] = None, #カタカナが多い方を削除
                 kanji_ratio_threshold: Optional[float] = None,
                 *args: Any,
                 **kwargs: Any) -> None: #漢字が多い方を削除
        super().__init__(*args, **kwargs)
        self.sentence_length_threshold = sentence_length_threshold
        self.comma_ratio_threshold = comma_ratio_threshold
        self.hiragana_ratio_threshold = hiragana_ratio_threshold
        self.katakana_ratio_threshold = katakana_ratio_threshold
        self.kanji_ratio_threshold = kanji_ratio_threshold

        self.hiragana_chars = set(chr(i) for i in range(12353, 12436))
        self.katakana_chars = set(chr(i) for i in range(12449, 12534))

    def apply(self, doc: Document) -> Document:
        normalized_doc = doc.text.replace('|', '')
        normalized_doc = re.sub(r'\【.*?\】', '', normalized_doc)
        sentences = re.split(r'[\n　.。 ]', normalized_doc)

        filtered_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if self.sentence_length_threshold != None:
                if len(sentence) <= self.sentence_length_threshold:
                    continue

            if self.comma_ratio_threshold != None:
                comma_count = sentence.count('、') + sentence.count(',')
                comma_ratio = comma_count / len(sentence)
                if comma_ratio > self.comma_ratio_threshold:
                    continue

            if self.hiragana_ratio_threshold != None:
                hiragana_count = sum(1 for char in sentence if char in self.hiragana_chars)
                hiragana_ratio = hiragana_count / len(sentence) 
                if hiragana_ratio < self.hiragana_ratio_threshold:
                    continue     
            
            if self.katakana_ratio_threshold != None:
                katakana_count = sum(1 for char in sentence if char in self.katakana_chars)
                katakana_ratio = katakana_count / len(sentence)
                if katakana_ratio > self.katakana_ratio_threshold:
                    continue

            if self.kanji_ratio_threshold != None:
                # 漢字の正規表現パターン
                kanji_pattern = re.compile(r'[\u4e00-\u9fff]')

                # 漢字の数をカウント
                kanji_count = len(kanji_pattern.findall(sentence))

                # 漢字の割合を計算
                kanji_ratio = kanji_count / len(sentence)
                if kanji_ratio > self.kanji_ratio_threshold:
                    continue
            
            filtered_sentences.append(sentence)

        doc.text = '。'.join(filtered_sentences)
        if doc.text != "":
            doc.text += "。" 
        else:
            doc.is_rejected = True
        return doc
    
class DiscardAdultContentWithEmbedding(Filter):
    """
    TokenFilter の実装例です.
    DiscardWithCharacterRatioの後に適用することを想定しています。
    日本語の成人向けコンテンツをEmbeddingを用いて排除します.
    """

    def __init__(
        self,
        adult_threshold: float = 0.25,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_path = "../../fastText/cc.ja.300.bin"
        self.adult_embedding_path = "./preprocessing/adult_embedding_avg.npy"
        self.adult_embedding = np.load(self.adult_embedding_path)
        self.adult_threshold = adult_threshold
        self.model = FastText.load_fasttext_format(self.model_path)

    def calculate_inappropriate_score(self, sentence):
        text_owakati = tagger.parse(sentence).split()
        word_scores = []
        for word in text_owakati:
            word_embedding = self.model.wv[word]
            similarity = np.dot(word_embedding, self.adult_embedding)
            word_scores.append(similarity)
        
        return np.nanmean(word_scores)

    def apply(self, doc: Document) -> Document:
        sentences = re.split('。', doc.text)

        filtered_sentences = []

        for sentence in sentences:
            if len(sentence) == 0:
                continue
            if self.calculate_inappropriate_score(sentence) <= self.adult_threshold:
                filtered_sentences.append(sentence)

        doc.text = '。'.join(filtered_sentences)
        if doc.text != "":
            doc.text += "。" 
        else:
            doc.is_rejected = True
        return doc