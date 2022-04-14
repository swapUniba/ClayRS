from typing import List

import gensim
from gensim.corpora import Dictionary
from gensim.models import LsiModel

from orange_cb_recsys.content_analyzer.embeddings.embedding_learner.embedding_learner import \
    GensimDocumentEmbeddingLearner
from orange_cb_recsys.utils.check_tokenization import check_tokenized


class GensimLatentSemanticAnalysis(GensimDocumentEmbeddingLearner):
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
        return "< GensimLatentSemanticAnalysis : model = " + str(self.model) + " >"
