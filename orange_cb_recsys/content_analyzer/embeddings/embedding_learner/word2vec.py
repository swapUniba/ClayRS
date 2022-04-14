from typing import List

from gensim.models import Word2Vec

from orange_cb_recsys.content_analyzer.embeddings.embedding_learner.embedding_learner import GensimWordEmbeddingLearner


class GensimWord2Vec(GensimWordEmbeddingLearner):
    """"
    Class that implements the Abstract Class GensimWordEmbeddingLearner
    Implementation of Word2Vec using the Gensim library
    """

    def __init__(self, reference: str = None, auto_save: bool = True, **kwargs):
        super().__init__(reference, auto_save, ".kv", **kwargs)

    def fit_model(self, corpus: List):
        self.model = Word2Vec(sentences=corpus, **self.additional_parameters).wv

    def __str__(self):
        return "GensimWord2Vec"

    def __repr__(self):
        return "< GensimWord2Vec : model = " + str(self.model) + " >"
