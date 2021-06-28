from typing import List

from gensim.models import RpModel
from gensim.corpora import Dictionary

from orange_cb_recsys.content_analyzer.embedding_learner.embedding_learner import GensimProjectionsWordEmbeddingLearner


class GensimRandomIndexing(GensimProjectionsWordEmbeddingLearner):
    """
    Class that implements the Abstract Class GensimProjectionsWordEmbeddingLearner
    Class that implements the random indexing using Gensim
    """

    def __init__(self, reference: str = None, auto_save: bool = True, **kwargs):
        super().__init__(reference, auto_save, ".model", **kwargs)

    def fit_model(self, corpus: List):
        """
        This method creates the model, using Gensim Random Projection.
        The model isn't then returned, but gets stored in the 'model' class attribute.
        """
        dictionary = Dictionary(corpus)
        self.model = RpModel(corpus, id2word=dictionary, **self.additional_parameters)

    def load_model(self):
        return RpModel.load(self.reference)

    def __str__(self):
        return "GensimRandomProjections"

    def __repr__(self):
        return "< GensimRandomProjections: model = " + str(self.model) + " >"
