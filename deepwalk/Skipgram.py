"""
skipgram算法
"""
from collections import Counter, Mapping
from concurrent.futures import ProcessPoolExecutor
from commonUtils.Loggings import Logger
from multiprocessing import cpu_count
# string_types是什么

from gensim.models import word2vec
from gensim.models.word2vec import Vocab


class Skipgram(word2vec):
    """A subclass to allow more customization of the Word2Vec internals."""

    def __init__(self, vocabulary_counts=None, **kwargs):
        self.vocabulary_counts = None

        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["workers"] = kwargs.get("workers", cpu_count())
        kwargs["size"] = kwargs.get("size", 128)
        kwargs["sentences"] = kwargs.get("sentences", None)
        kwargs["window"] = kwargs.get("window", 10)
        kwargs["sg"] = 1
        kwargs["hs"] = 1

        if vocabulary_counts != None:
            self.vocabulary_counts = vocabulary_counts
        super(Skipgram, self).__init__(**kwargs)
