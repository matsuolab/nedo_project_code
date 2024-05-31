from .Text2Vec import extract_nouns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
# カスタム変換器の定義


class TextTruncator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # 特に何もしない
        return self

    def transform(self, X):
        # 各テキストの最初の100文字を取り出す
        return [text[:100] for text in X]


def prepare_pileline(n_clusters=3):

    # パイプラインの作成
    pipeline = Pipeline([
        ('truncator', TextTruncator()),  # テキストを100文字に切り詰めるステップ
        ('tfidf', TfidfVectorizer(tokenizer=extract_nouns, norm=None)),
        # ('cluster', MiniBatchKMeans(n_clusters=n_clusters, random_state=0))
        ('cluster',  DBSCAN(eps=1, min_samples=10))
    ])

    return pipeline
