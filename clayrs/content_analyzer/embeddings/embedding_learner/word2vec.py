from typing import List

from gensim.models import Word2Vec

from clayrs.content_analyzer.embeddings.embedding_learner.embedding_learner import GensimWordEmbeddingLearner


class GensimWord2Vec(GensimWordEmbeddingLearner):
    """
    Class that implements Word2Vec model thanks to the Gensim library.

    If a pre-trained local Word2Vec model must be loaded, put its path in the `reference` parameter.
    Otherwise, a Word2Vec model will be trained from scratch based on the preprocessed corpus of the contents to complexly
    represent

    If you'd like to save the model once trained, set the path in the `reference` parameter and set
    `auto_save=True`. If `reference` is None, trained model won't be saved after training and will only be used to
    produce contents in the current run

    Additional parameters regarding the model itself could be passed, check [gensim documentation](https://radimrehurek.com/gensim/models/word2vec.html)
    to see what else can be customized

    Args:
        reference: Path of the model to load/where the model trained will be saved if `auto_save=True`. If None the
            trained model won't be saved after training and will only be used to produce contents in the current run
        auto_save: If True, the model will be saved in the path specified in `reference` parameter
    """

    def __init__(self, reference: str = None, auto_save: bool = True, **kwargs):
        super().__init__(reference, auto_save, ".kv", **kwargs)

    def fit_model(self, corpus: List):
        self.model = Word2Vec(sentences=corpus, **self.additional_parameters).wv

    def __str__(self):
        return "GensimWord2Vec"

    def __repr__(self):
        return f'GensimWord2Vec(attributes={str(self.model)})'
