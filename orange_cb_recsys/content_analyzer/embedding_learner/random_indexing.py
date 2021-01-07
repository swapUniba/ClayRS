from typing import List

from gensim.models import RpModel
from gensim.corpora import Dictionary

from orange_cb_recsys.content_analyzer.embedding_learner.embedding_learner import EmbeddingLearner
from orange_cb_recsys.content_analyzer.information_processor.\
    information_processor import TextProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource


class GensimRandomIndexing(EmbeddingLearner):
    """
    Class that implements the random indexing using Gensim

    Args:
        source (RawInformationSource): Source where the content is stored.
        preprocessor (InformationProcessor): Instance of the class InformationProcessor,
            specify how to process (can be None) the source data, before
            use it for model computation
        field_list (List<str>): Field name list.
    """
    def __init__(self, source: RawInformationSource,
                 preprocessor: TextProcessor,
                 field_list: List[str]):
        super().__init__(source, preprocessor, field_list)

    def fit(self):
        """
        This method creates the model, using Gensim Random Projection.
        The model isn't then returned, but gets stored in the 'model' class attribute.
        """
        corpus = self.extract_corpus()
        dictionary = Dictionary(corpus)
        model = RpModel(corpus, id2word=dictionary)
        self.model = model
