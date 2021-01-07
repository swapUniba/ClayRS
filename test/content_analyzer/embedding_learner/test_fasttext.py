from unittest import TestCase

from orange_cb_recsys.content_analyzer.embedding_learner.fasttext import GensimFastText
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile


class TestGensimFastText(TestCase):
    def test_fit(self):
        field_list = ['Title', 'Year', 'Genre']

        file_path = '../../../datasets/movies_info_reduced.json'
        try:
            with open(file_path):
                pass
        except FileNotFoundError:
            file_path = 'datasets/movies_info_reduced.json'

        GensimFastText(source=JSONFile(file_path),
                       preprocessor=NLTK(),
                       field_list=field_list).fit()


