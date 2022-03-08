from orange_cb_recsys.content_analyzer.content_representation.content import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    SynsetDocumentFrequency
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized
from typing import List, Union

from collections import Counter


class PyWSDSynsetDocumentFrequency(SynsetDocumentFrequency):
    """
    Pywsd word sense disambiguation
    """
    def __init__(self):
        from pywsd import disambiguate

        self.disambiguate = disambiguate
        super().__init__()

    def produce_single_repr(self, field_data: Union[List[str], str]) -> FeaturesBagField:
        """
        Produces a bag of features whose key is a wordnet synset and whose value is the frequency of the synset in the
        field data text
        """

        field_data = check_not_tokenized(field_data)

        synsets = self.disambiguate(field_data)
        synsets = [synset for word, synset in synsets if synset is not None]

        return FeaturesBagField(Counter(synsets))

    def __str__(self):
        return "PyWSDSynsetDocumentFrequency"

    def __repr__(self):
        return f'PyWSDSynsetDocumentFrequency(disambiguate={self.disambiguate}'
