from unittest import TestCase

from clayrs.content_analyzer.information_processor.spacy_processor import Spacy


class TestSpacy(TestCase):

    def test_process_stopwords(self):
        spa = Spacy(stopwords_removal=True)
        expected = ["striped", "bats", "hanging", "feet", "best"]
        result = spa.process("The striped bats are hanging on their feet for the best")
        self.assertEqual(expected, result)

        # add new stopwords
        spa = Spacy(stopwords_removal=True, new_stopwords=['bats', 'best'])
        expected = ["striped", "hanging", "feet"]
        result = spa.process("The striped bats are hanging on their feet for the best")
        self.assertEqual(expected, result)

        # allow some stopwords
        spa = Spacy(stopwords_removal=True, not_stopwords=['The', 'the', 'on'])
        expected = ["The", "striped", "bats", "hanging", "on", "feet", "the", "best"]
        result = spa.process("The striped bats are hanging on their feet for the best")
        self.assertEqual(expected, result)

        # add new stopwords and allow some stopwords
        spa = Spacy(stopwords_removal=True, new_stopwords=['bats', 'best'], not_stopwords=['The', 'the', 'on'])
        expected = ["The", "striped", "hanging", "on", "feet", "the"]
        result = spa.process("The striped bats are hanging on their feet for the best")
        self.assertEqual(expected, result)

    def test_process_lemmatization(self):
        spa = Spacy(lemmatization=True)
        expected = ["the", "stripe", "bat", "be", "hang", "on", "their", "foot", "for", "good"]
        result = spa.process("The striped bats are hanging on their feet for best")
        self.assertEqual(expected, result)

    def test_remove_punctuation(self):
        spa = Spacy(remove_punctuation=True)
        expected = ["Hello", "there", "How", "are", "you", "I", "'m", "fine", "thanks"]
        result = spa.process("Hello there. How are you? I'm fine, thanks.")
        self.assertEqual(expected, result)

    def test_url_tagging(self):
        spa = Spacy(url_tagging=True)
        expected = ["The", "striped", "<URL>", "bats", "<URL>", "are", "<URL>", "hanging", "on", "their", "feet",
                    "for", "best", "<URL>"]
        result = spa.process("The striped http://facebook.com bats https://github.com are "
                             "http://facebook.com hanging on their feet for best http://twitter.it")
        self.assertEqual(expected, result)

    def test_entity_recognition(self):
        spa = Spacy(named_entity_recognition=True)
        expected = ["Facebook", "was", "fined", "by", "<Hewlett_ORG_B>", "<Packard_ORG_I>", "for", "spending",
                    "<100_MONEY_B>", "<€_MONEY_I>", "to", "buy", "<Cristiano_PERSON_B>", "<Ronaldo_PERSON_I>",
                    "from", "<Juventus_ORG_B>"]
        result = spa.process(
            "Facebook was fined by Hewlett Packard for spending 100€ to buy Cristiano Ronaldo from "
            "Juventus")

        self.assertEqual(expected, result)

    def test_multiple_operations(self):
        # Test for lemmatization with multiple whitespaces removal
        spa = Spacy(strip_multiple_whitespaces=True, lemmatization=True)
        expected = ["the", "stripe", "bat", "be", "hang", "on", "their", "foot", "for", "good"]
        result = spa.process("The   striped  bats    are    hanging   on   their    feet   for  best")
        self.assertEqual(expected, result)

        # Test for lemmatization with multiple whitespaces removal and URL tagging
        spa = Spacy(strip_multiple_whitespaces=True, lemmatization=True, url_tagging=True)
        expected = ["the", "stripe", "<URL>", "bat", "<URL>", "be", "<URL>", "hang", "on", "their", "foot", "for",
                    "good", "<URL>"]
        result = spa.process(
            "The   striped http://facebook.com bats https://github.com   are   http://facebook.com "
            "hanging   on   their    feet   for  best  http://twitter.it")

        self.assertEqual(expected, result)

        # Test for lemmatization, multiple whitespaces removal, URL tagging and stopwords removal with new stopwords
        spa = Spacy(lemmatization=True, strip_multiple_whitespaces=True, url_tagging=True, stopwords_removal=True,
                    new_stopwords=['feet'])
        expected = ["stripe", "<URL>", "bat", "<URL>", "<URL>", "hang", "good", "<URL>"]
        result = spa.process(
            "The   striped http://facebook.com bats https://github.com   are   http://facebook.com "
            "hanging   on   their    feet   for  best  http://twitter.it")
        self.assertEqual(expected, result)

        # Test for lemmatization, multiple whitespaces removal, URL tagging, stemming, stopwords removal with new
        # stopwords and remove punctuation
        spa = Spacy(lemmatization=True, strip_multiple_whitespaces=True, url_tagging=True, stopwords_removal=True,
                    new_stopwords=['feet'], remove_punctuation=True)
        expected = ["stripe", "<URL>", "bat", "<URL>", "<URL>", "hang", "good", "<URL>"]
        result = spa.process(
            "The   striped http://facebook.com bats https://github.com   are.   http://facebook.com hanging   On   "
            "their.    feet;   for:  best  http://twitter.it")

        self.assertEqual(expected, result)
