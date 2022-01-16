from unittest import TestCase

from orange_cb_recsys.content_analyzer.information_processor.spacy import Spacy


class TestSpacy(TestCase):

    def test_process(self):
        #Test for only stop words removal
        spacy1 = Spacy(stopwords_removal=True, url_tagging=True)
        spacy1.set_lang("")
        self.assertEqual(spacy1.process(
                "The boys went to the mountains and tried skiing"),
                ["boys", "went", "mountains", "tried", "skiing"])

        spacy1.stopwords_removal = False


        #Test for only lemmatization
        spacy1.lemmatization  = True
        self.assertEqual(spacy1.process(
                "The boys went to the mountains and tried skiing"),
                ["the", "boy", "go", "to","the", "mountain", "and", "try", "ski"])

        spacy1.strip_multiple_whitespaces = True
        self.assertEqual(spacy1.process(
            "The boys    went to the mountains   and tried skiing"),
                ["the", "boy", "go", "to","the", "mountain", "and", "try", "ski"])

        # Test for lemmatization with multiple whitespaces removal and URL tagging
        spacy1.url_tagging = True
        self.assertEqual(spacy1.process(
            "The boys   https://www.google.com went to the mountains   and tried skiing"),
            ["the", "boy", "<URL>", "go", "to", "the", "mountain", "and", "try", "ski"])

        spacy1.stopwords_removal = True
        self.assertEqual(spacy1.process(
            "The boys   https://www.google.com went to the mountains   and tried skiing"),
            ["boy", "<URL>", "go", "mountain", "try", "ski"])
