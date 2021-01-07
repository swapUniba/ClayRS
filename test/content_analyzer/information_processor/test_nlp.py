from unittest import TestCase

from nltk import Tree

from orange_cb_recsys.content_analyzer.information_processor.nlp import NLTK


class TestNLTK(TestCase):
    """nltk = NLTK(stopwords_removal=True,
                stemming=True,
                lemmatization=True,
                named_entity_recognition=True,
                strip_multiple_whitespaces=True,
                url_tagging=True,
                lan="english")"""

    def test_process(self):
        #Test for only stop words removal
        nltka = NLTK(stopwords_removal=True, url_tagging=True)
        nltka.set_lang("")
        self.assertEqual(nltka.process(
                "The striped bats are hanging on their feet for the best"),
                ["striped", "bats", "hanging", "feet", "best"])

        #Test for only stemming
        nltka.stemming = True
        nltka.stopwords_removal = False
        self.assertEqual(nltka.process(
                "My name is Francesco and I am a student at the University of the city of Bari"),
                ["my", "name", "is", "francesco", "and", "i", "am", "a", "student", "at", "the", "univers", "of", "the", "citi", "of", "bari"])
        nltka.stemming = False

        #Test for only lemmatization
        nltka.lemmatization  = True
        self.assertEqual(nltka.process(
                "The striped bats are hanging on their feet for best"),
                ["The", "strip", "bat", "be", "hang", "on", "their", "foot", "for", "best"])

        #Test for lemmatization with multiple whitespaces removal
        nltka.strip_multiple_whitespaces = True
        self.assertEqual(nltka.process(
                "The   striped  bats    are    hanging   on   their    feet   for  best"),
                ["The", "strip", "bat", "be", "hang", "on", "their", "foot", "for", "best"])

        #Test for lemmatization with multiple whitespaces removal and URL tagging
        nltka.url_tagging = True
        self.assertEqual(nltka.process(
                "The   striped http://facebook.com bats https://github.com   are   http://facebook.com hanging   on   their    feet   for  best  http://twitter.it"),
                ["The", "strip", "<URL>", "bat", "<URL>", "be", "<URL>", "hang", "on", "their", "foot", "for", "best", "<URL>"])

        # Test for lemmatization, multiple whitespaces removal, URL tagging and stemming
        nltka.stemming = True
        self.assertEqual(nltka.process(
            "The   striped http://facebook.com bats https://github.com   are   http://facebook.com hanging   on   their    feet   for  best  http://twitter.it"),
            ["the", "strip", "<url>", "bat", "<url>", "be", "<url>", "hang", "on", "their", "foot", "for", "best", "<url>"])

        # Test for lemmatization, multiple whitespaces removal, URL tagging, stemming, stop words removal
        nltka.stopwords_removal = True
        self.assertEqual(nltka.process(
            "The   striped http://facebook.com bats https://github.com   are   http://facebook.com hanging   on   their    feet   for  best  http://twitter.it"),
            ["strip", "<url>", "bat", "<url>", "<url>", "hang", "foot", "best", "<url>"])

        nltka.named_entity_recognition = True
        nltka.stopwords_removal = False
        nltka.stemming = False
        nltka.lemmatization = False
        result = nltka.process("Facebook was fined by Hewlett Packard for spending 100€ to buy Cristiano Ronaldo from Juventus")

        self.assertEqual(result,
                         Tree('S', [Tree('PERSON', [('Facebook', 'NNP')]), ('was', 'VBD'), ('fined', 'VBN'), ('by', 'IN'), Tree('PERSON', [('Hewlett', 'NNP'), ('Packard', 'NNP')]), ('for', 'IN'), ('spending', 'VBG'), ('100€', 'CD'), ('to', 'TO'), ('buy', 'VB'), Tree('PERSON', [('Cristiano', 'NNP'), ('Ronaldo', 'NNP')]), ('from', 'IN'), Tree('GPE', [('Juventus', 'NNP')])]))
