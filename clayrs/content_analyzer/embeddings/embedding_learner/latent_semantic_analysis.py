from typing import List

import gensim
from gensim.corpora import Dictionary
from gensim.models import LsiModel

from clayrs.content_analyzer.embeddings.embedding_learner.embedding_learner import GensimDocumentEmbeddingLearner
from clayrs.content_analyzer.utils.check_tokenization import check_tokenized


class GensimLatentSemanticAnalysis(GensimDocumentEmbeddingLearner):
    """
    Class that implements Latent Semantic Analysis (A.K.A. Latent Semantic Indexing)
    (LSI) thanks to the Gensim library.

    If a pre-trained local Word2Vec model must be loaded, put its path in the `reference` parameter.
    Otherwise, a Word2Vec model will be trained from scratch based on the preprocessed corpus of the contents to complexly
    represent

    If you'd like to save the model once trained, set the path in the `reference` parameter and set
    `auto_save=True`. If `reference` is None, trained model won't be saved after training and will only be used to
    produce contents in the current run

    Additional parameters regarding the model itself could be passed, check [gensim documentation](https://radimrehurek.com/gensim/models/lsimodel.html)
    to see what else can be customized

    Args:
        reference: Path of the model to load/where the model trained will be saved if `auto_save=True`. If None the
            trained model won't be saved after training and will only be used to produce contents in the current run
        auto_save: If True, the model will be saved in the path specified in `reference` parameter
    """

    def __init__(self, reference: str = None, auto_save: bool = True,  **kwargs):
        super().__init__(reference, auto_save, ".model", **kwargs)

    def fit_model(self, corpus: List):
        dictionary = Dictionary(corpus)
        word_docs_matrix = [dictionary.doc2bow(doc) for doc in corpus]
        self.model = LsiModel(word_docs_matrix, id2word=dictionary, **self.additional_parameters)

    def load_model(self):
        return LsiModel.load(self.reference)

    def get_vector_size(self) -> int:
        return self.model.num_topics

    def get_embedding(self, document_tokenized: List[str]):
        unseen_doc = self.model.id2word.doc2bow(check_tokenized(document_tokenized))

        # if document is totally new (no word in train corpus) KeyError is raised
        # and load method of embedding source will fill the document vector with zeros
        if len(unseen_doc) == 0:
            raise KeyError

        sparse_vector = self.model[unseen_doc]
        dense_vector = gensim.matutils.sparse2full(sparse_vector, self.model.num_topics)
        return dense_vector

    def __str__(self):
        return "GensimLatentSemanticAnalysis"

    def __repr__(self):
        return f"GensimLatentSemanticAnalysis(reference={self.reference}, auto_save={self._auto_save}, " \
               f"{', '.join(f'{arg}={val}' for arg, val in self._additional_parameters.items())})"
