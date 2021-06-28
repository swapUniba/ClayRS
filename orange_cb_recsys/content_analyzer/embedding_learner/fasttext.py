from typing import List
from gensim.models.fasttext import FastText, save_facebook_model, load_facebook_vectors

from orange_cb_recsys.content_analyzer.embedding_learner.embedding_learner import GensimWordEmbeddingLearner


class GensimFastText(GensimWordEmbeddingLearner):
    """"
    Class that implements the Abstract Class GensimWordEmdeddingLearner
    Implementation of FastText using the Gensim library.
    """

    def __init__(self, reference: str = None, auto_save: bool = True, **kwargs):
        super().__init__(reference, auto_save, ".bin", **kwargs)

    def fit_model(self, corpus: List):
        self.model = FastText(sentences=corpus, **self.additional_parameters)

    def load_model(self):
        return load_facebook_vectors(self.reference)

    def save(self):
        save_facebook_model(self.model, self.reference)

    def __str__(self):
        return "FastText"

    def __repr__(self):
        return "< FastText : model = " + str(self.model) + " >"
