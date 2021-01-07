from typing import List

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from orange_cb_recsys.content_analyzer.embedding_learner.embedding_learner import EmbeddingLearner
from orange_cb_recsys.content_analyzer.information_processor.\
    information_processor import TextProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource


class GensimDoc2Vec(EmbeddingLearner):
    """"
    Class that implements the Abstract Class EmbeddingLearner.
    Implementation of Word2Vec using the Gensim library.

    Args:
        source (RawInformationSource): Source where the content is stored.
        preprocessor (InformationProcessor): Instance of the class InformationProcessor,
        specify how to process (can be None) the source data, before
        use it for model computation
        field_list (List<str>): Field name list.
    """
    def __init__(self, source: RawInformationSource,
                 preprocessor: TextProcessor,
                 field_list: List[str],
                 **kwargs):
        super().__init__(source, preprocessor, field_list)

        if "max_epochs" in kwargs.keys():
            self.__max_epochs = kwargs["max_epochs"]
        else:
            self.__max_epochs = 100

        if "vec_size" in kwargs.keys():
            self.__vec_size = kwargs["vec_size"]
        else:
            self.__vec_size = 20

        if "alpha" in kwargs.keys():
            self.__alpha = kwargs["alpha"]
        else:
            self.__alpha = 0.025

        self.__model = None

    def __str__(self):
        return "GensimDoc2Vec"

    def __repr__(self):
        return "< GensimDoc2Vec :" + \
               "loader = " + str(self.__source) + \
               "preprocessor = " + str(self.__preprocessor) + " >"

    def fit(self):
        """
        This method creates the model, using Gensim Doc2Vec.
        The model isn't then returned, but gets stored in the 'model' class attribute.
        """
        corpus = self.extract_corpus()
        tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(corpus)]
        model = Doc2Vec(vector_size=self.__vec_size,
                        alpha=self.__alpha,
                        min_alpha=0.00025,
                        min_count=1,
                        dm=1)

        model.build_vocab(tagged_data)  # this creates the vocabulary

        for epoch in range(self.__max_epochs):
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            model.alpha -= 0.0002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        self.model = model
