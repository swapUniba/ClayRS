from typing import List

from gensim.corpora import Dictionary
from gensim.models import LsiModel

from orange_cb_recsys.content_analyzer.embedding_learner.embedding_learner import GensimProjectionsWordEmbeddingLearner


class GensimLatentSemanticAnalysis(GensimProjectionsWordEmbeddingLearner):
    """
    Class that implements the Abstract Class GensimProjectionsWordEmbeddingLearner
    Class that implements latent semantic analysis using Gensim
    """

    def __init__(self, reference: str = None, auto_save: bool = True,  **kwargs):
        super().__init__(reference, auto_save, ".model", **kwargs)

    def fit_model(self, corpus: List):
        """
        This method creates the model, using Gensim Latent Semantic Analysis.
        The model isn't then returned, but gets stored in the 'model' class attribute.
        """
        dictionary = Dictionary(corpus)
        word_docs_matrix = [dictionary.doc2bow(doc) for doc in corpus]
        self.model = LsiModel(word_docs_matrix, id2word=dictionary, **self.additional_parameters)

    def load_model(self):
        return LsiModel.load(self.reference)

    def get_vector_size(self) -> int:
        return len(self.model.get_topics())

    def __str__(self):
        return "GensimLatentSemanticAnalysis"

    def __repr__(self):
        return "< GensimLatentSemanticAnalysis : model = " + str(self.model) + " >"
