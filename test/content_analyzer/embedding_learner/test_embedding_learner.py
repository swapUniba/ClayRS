from unittest import TestCase

import os

from orange_cb_recsys.content_analyzer.embedding_learner.latent_semantic_analysis import GensimLatentSemanticAnalysis
from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile


class TestEmbeddingLearner(TestCase):
    def test_extract_corpus(self):
        preprocessor = NLTK(stopwords_removal=True, stemming=True)
        fields = ["Title", "Released"]
        expected = [['jumanji', '15', 'dec', '1995'],
                    ['grumpier', 'old', 'men', '22', 'dec', '1995'],
                    ['toy', 'stori', '22', 'nov', '1995'],
                    ['father', 'bride', 'part', 'ii', '08', 'dec', '1995'],
                    ['heat', '15', 'dec', '1995'],
                    ['tom', 'huck', '22', 'dec', '1995'],
                    ['wait', 'exhal', '22', 'dec', '1995'],
                    ['sabrina', '15', 'dec', '1995'],
                    ['dracula', ':', 'dead', 'love', '22', 'dec', '1995'],
                    ['nixon', '05', 'jan', '1996'],
                    ['american', 'presid', '17', 'nov', '1995'],
                    ['goldeney', '17', 'nov', '1995'],
                    ['balto', '22', 'dec', '1995'],
                    ['cutthroat', 'island', '22', 'dec', '1995'],
                    ['casino', '22', 'nov', '1995'],
                    ['sudden', 'death', '22', 'dec', '1995'],
                    ['sens', 'sensibl', '26', 'jan', '1996'],
                    ['four', 'room', '25', 'dec', '1995'],
                    ['money', 'train', '22', 'nov', '1995'],
                    ['ace', 'ventura', ':', 'natur', 'call', '10', 'nov', '1995']]

        file_path = "datasets/movies_info_reduced.json"
        try:
            with open(file_path):
                pass
        except FileNotFoundError:
            file_path = "../../../datasets/movies_info_reduced.json"

        src = JSONFile(file_path)
        learner = GensimLatentSemanticAnalysis(src, preprocessor, fields)
        generated = learner.extract_corpus()

        self.assertEqual(generated, expected)

    def test_save(self):
        self.skipTest("_")
        preprocessor = NLTK(stopwords_removal=True)
        fields = ["Plot"]
        try:
            src = JSONFile("datasets/movies_info_reduced.json")
            learner = GensimLatentSemanticAnalysis(src, preprocessor, fields)
            learner.fit()
        except FileNotFoundError:
            src = JSONFile("../../../datasets/movies_info_reduced.json")
            learner = GensimLatentSemanticAnalysis(src, preprocessor, fields)
            learner.fit()
        learner.save()



