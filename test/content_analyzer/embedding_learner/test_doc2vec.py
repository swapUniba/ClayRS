from unittest import TestCase

from orange_cb_recsys.content_analyzer.embedding_learner.doc2vec import GensimDoc2Vec
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile


class TestGensimDoc2Vec(TestCase):
    def test_fit(self):

        try:
            path = "datasets/d2v_test_data.json"
            with open(path):
                pass
        except FileNotFoundError:
            path = "../../../datasets/d2v_test_data.json"

        GensimDoc2Vec(source=JSONFile(file_path=path), preprocessor=NLTK(), field_list=["doc_field"]).fit()
