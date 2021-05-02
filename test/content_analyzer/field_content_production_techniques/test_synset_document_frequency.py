from unittest import TestCase

from orange_cb_recsys.content_analyzer.field_content_production_techniques import SynsetDocumentFrequency
from nltk.corpus import wordnet as wn


class TestSynsetDocumentFrequency(TestCase):
    def test_produce_content(self):
        technique = SynsetDocumentFrequency()
        example_text = "Hello, this is beautiful. And also beautiful"

        syn = technique.produce_content(example_text)

        result = sorted(syn.value.elements())

        expected = [wn.synset('beautiful.s.02'), wn.synset('beautiful.s.02'),
                    wn.synset('besides.r.02'), wn.synset('hello.n.01')]

        self.assertEqual(result, expected)
