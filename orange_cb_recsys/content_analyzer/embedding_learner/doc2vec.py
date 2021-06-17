from typing import List

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from orange_cb_recsys.content_analyzer.embedding_learner.embedding_learner import GensimWordEmbeddingLearner


class GensimDoc2Vec(GensimWordEmbeddingLearner):
    """"
    Class that implements the Abstract Class GensimWordEmbeddingLearner
    Implementation of Doc2Vec using the Gensim library.
    """

    def __init__(self, reference: str = None, auto_save: bool = True, **kwargs):
        super().__init__(reference, auto_save, ".model", **kwargs)

    def fit_model(self, corpus: List):
        tagged_data = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
        self.model = Doc2Vec(tagged_data, **self.additional_parameters)

    def load_model(self):
        return Doc2Vec.load(self.reference)

    def __str__(self):
        return "GensimDoc2Vec"

    def __repr__(self):
        return "< GensimDoc2Vec : model = " + str(self.model) + " >"
