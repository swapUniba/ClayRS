import os
import time
from typing import List

from gensim.models import Word2Vec

from orange_cb_recsys.content_analyzer.embedding_learner.embedding_learner import EmbeddingLearner
from orange_cb_recsys.content_analyzer.information_processor\
    .information_processor import TextProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.const import DEVELOPING, home_path


class GensimWord2Vec(EmbeddingLearner):
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

        self.optionals = {}
        if "corpus_file" in kwargs.keys():
            self.optionals["corpus_file"] = kwargs["corpus_file"]

        if "size" in kwargs.keys():
            self.optionals["size"] = kwargs["size"]

        if "window" in kwargs.keys():
            self.optionals["window"] = kwargs["window"]

        if "min_count" in kwargs.keys():
            self.optionals["min_count"] = kwargs["min_count"]

        if "workers" in kwargs.keys():
            self.optionals["workers"] = kwargs["workers"]

        if "alpha" in kwargs.keys():
            self.optionals["alpha"] = kwargs["alpha"]

        if "min_alpha" in kwargs.keys():
            self.optionals["min_alpha"] = kwargs["min_alpha"]

        if "sg" in kwargs.keys():
            self.optionals["sg"] = kwargs["sg"]

        if "hs" in kwargs.keys():
            self.optionals["hs"] = kwargs["hs"]

        if "seed" in kwargs.keys():
            self.optionals["seed"] = kwargs["seed"]

        if "max_vocab_size" in kwargs.keys():
            self.optionals["max_vocab_size"] = kwargs["max_vocab_size"]

        if "sample" in kwargs.keys():
            self.optionals["sample"] = kwargs["sample"]

        if "negative" in kwargs.keys():
            self.optionals["negative"] = kwargs["negative"]

        if "ns_exponent" in kwargs.keys():
            self.optionals["ns_exponent"] = kwargs["ns_exponent"]

        if "cbow_mean" in kwargs.keys():
            self.optionals["cbow_mean"] = kwargs["cbow_mean"]

        if "hashfxn" in kwargs.keys():
            self.optionals["hashfxn"] = kwargs["hashfxn"]

        if "iter" in kwargs.keys():
            self.optionals["iter"] = kwargs["iter"]

        if "trim_rule" in kwargs.keys():
            self.optionals["trim_rule"] = kwargs["trim_rule"]

        if "batch_words" in kwargs.keys():
            self.optionals["batch_words"] = kwargs["batch_words"]

        if "min_n" in kwargs.keys():
            self.optionals["min_n"] = kwargs["min_n"]

        if "max_n" in kwargs.keys():
            self.optionals["max_n"] = kwargs["max_n"]

        if "word_ngrams" in kwargs.keys():
            self.optionals["word_ngrams"] = kwargs["word_ngrams"]

        if "bucket" in kwargs.keys():
            self.optionals["bucket"] = kwargs["bucket"]

        if "callbacks" in kwargs.keys():
            self.optionals["callbacks"] = kwargs["callbacks"]

        if "max_final_vocab" in kwargs.keys():
            self.optionals["max_final_vocab"] = kwargs["max_final_vocab"]

        if "compute_loss" in kwargs.keys():
            self.optionals["compute_loss"] = kwargs["compute_loss"]

        if "compatible_hash" in kwargs.keys():
            self.optionals["compatible_hash"] = kwargs["compatible_hash"]

        if "sorted_vocab" in kwargs.keys():
            self.optionals["sorted_vocab"] = kwargs["sorted_vocab"]

        if "ephocs" in kwargs.keys():
            self.__epochs = kwargs["ephocs"]
        else:
            self.__epochs = 50

    def __str__(self):
        return "GensimWord2Vec"

    def __repr__(self):
        return "< GensimWord2Vec :" + \
               "source = " + str(self.source) + \
               "preprocessor = " + str(self.preprocessor) + " >"

    def fit(self):
        """
        This method creates the model, using Word 2 Vec.
        The model isn't then returned, but gets stored in the 'model' class attribute.
        """
        data_to_train = self.extract_corpus()
        model = Word2Vec(sentences=data_to_train, **self.optionals)
        # model.build_vocab(data_to_train)
        model.train(sentences=data_to_train,
                    total_examples=model.corpus_count,
                    epochs=self.__epochs)
        self.model = model

    def save(self):
        embeddings_path = './'
        if not DEVELOPING:
            embeddings_path = os.path.join(home_path, 'embeddings')
        self.model.wv.save_word2vec_format(embeddings_path + str(time.time()) + ".bin", binary=True)
