from unittest import TestCase

from nltk import Tree

from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK


class TestNLTK(TestCase):

    def test_process_stopwords(self):
        nltka = NLTK(stopwords_removal=True)
        expected = ["striped", "bats", "hanging", "feet", "best"]
        result = nltka.process("The striped bats are hanging on their feet for the best")
        self.assertEqual(expected, result)

    def test_process_stemming(self):
        # Test for only stemming
        nltka = NLTK(stemming=True)
        expected = ["my", "name", "is", "francesco", "and", "i", "am", "a", "student", "at", "the",
                    "univers", "of", "the", "citi", "of", "bari"]
        result = nltka.process("My name is Francesco and I am a student at the University of the city of Bari")

        self.assertEqual(expected, result)

    def test_process_lemmatization(self):
        nltka = NLTK(lemmatization=True)
        expected = ["The", "strip", "bat", "be", "hang", "on", "their", "foot", "for", "best"]
        result = nltka.process("The striped bats are hanging on their feet for best")
        self.assertEqual(expected, result)

    def test_remove_punctuation(self):
        nltka = NLTK(remove_punctuation=True)
        expected = ["Hello", "there", "How", "are", "you", "I", "m", "fine", "thanks"]
        result = nltka.process("Hello there. How are you? I'm fine, thanks.")
        self.assertEqual(expected, result)

    def test_url_tagging(self):
        nltka = NLTK(url_tagging=True)
        expected = ["The", "striped", "<URL>", "bats", "<URL>", "are", "<URL>", "hanging", "on", "their", "feet", "for",
                    "best", "<URL>"]
        result = nltka.process("The striped http://facebook.com bats https://github.com are "
                               "http://facebook.com hanging on their feet for best http://twitter.it")
        self.assertEqual(expected, result)

    def test_entity_recognition(self):
        nltka = NLTK(named_entity_recognition=True)
        expected = Tree('S',
                        [Tree('PERSON', [('Facebook', 'NNP')]), ('was', 'VBD'), ('fined', 'VBN'), ('by', 'IN'),
                         Tree('PERSON', [('Hewlett', 'NNP'), ('Packard', 'NNP')]), ('for', 'IN'),
                         ('spending', 'VBG'), ('100€', 'CD'), ('to', 'TO'), ('buy', 'VB'),
                         Tree('PERSON', [('Cristiano', 'NNP'), ('Ronaldo', 'NNP')]), ('from', 'IN'),
                         Tree('GPE', [('Juventus', 'NNP')])])
        result = nltka.process("Facebook was fined by Hewlett Packard for spending 100€ to buy Cristiano Ronaldo from "
                               "Juventus")

        self.assertEqual(expected, result)

    def test_multiple_operations(self):
        # Test for lemmatization with multiple whitespaces removal
        nltka = NLTK(strip_multiple_whitespaces=True, lemmatization=True)
        expected = ["The", "strip", "bat", "be", "hang", "on", "their", "foot", "for", "best"]
        result = nltka.process("The   striped  bats    are    hanging   on   their    feet   for  best")
        self.assertEqual(expected, result)

        # Test for lemmatization with multiple whitespaces removal and URL tagging
        nltka = NLTK(strip_multiple_whitespaces=True, lemmatization=True, url_tagging=True)
        expected = ["The", "strip", "<URL>", "bat", "<URL>", "be", "<URL>", "hang", "on", "their", "foot", "for",
                    "best", "<URL>"]
        result = nltka.process("The   striped http://facebook.com bats https://github.com   are   http://facebook.com "
                               "hanging   on   their    feet   for  best  http://twitter.it")

        self.assertEqual(expected, result)

        # Test for lemmatization, multiple whitespaces removal, URL tagging and stemming
        nltka = NLTK(lemmatization=True, strip_multiple_whitespaces=True, url_tagging=True, stemming=True)
        expected = ["the", "strip", "<url>", "bat", "<url>", "be", "<url>", "hang", "on", "their", "foot", "for",
                    "best", "<url>"]
        result = nltka.process("The   striped http://facebook.com bats https://github.com   are   http://facebook.com "
                               "hanging   on   their    feet   for  best  http://twitter.it")
        self.assertEqual(expected, result)

        # Test for lemmatization, multiple whitespaces removal, URL tagging, stemming, stop words removal,
        # remove punctuation
        nltka = NLTK(lemmatization=True, strip_multiple_whitespaces=True, url_tagging=True, stemming=True,
                     stopwords_removal=True, remove_punctuation=True)
        expected = ["strip", "<url>", "bat", "<url>", "<url>", "hang", "foot", "best", "<url>"]
        result = nltka.process(
            "The   striped, http://facebook.com bats https://github.com   are.   http://facebook.com hanging   On   "
            "their.    feet;   for:  best  http://twitter.it")

        self.assertEqual(expected, result)
