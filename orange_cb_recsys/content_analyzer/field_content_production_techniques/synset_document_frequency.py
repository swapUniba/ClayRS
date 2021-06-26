# import nltk
#
# try:
#     nltk.data.find('averaged_perceptron_tagger')
# except LookupError:
#     nltk.download('averaged_perceptron_tagger')
# try:
#     nltk.data.find('wordnet')
# except LookupError:
#     nltk.download('wordnet')
# try:
#     nltk.data.find('punkt')
# except LookupError:
#     nltk.download('punkt')

from orange_cb_recsys.content_analyzer.content_representation.content import FeaturesBagField
from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    SynsetDocumentFrequency
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized
from typing import List, Union
from pywsd import disambiguate
from collections import Counter


class PyWSDSynsetDocumentFrequency(SynsetDocumentFrequency):
    """
    Pywsd word sense disambiguation
    """
    def __init__(self):
        super().__init__()

    def produce_single_repr(self, field_data: Union[List[str], str]) -> FeaturesBagField:
        """
        Produces a bag of features whose key is a wordnet synset and whose value is the frequency of the synset in the
        field data text
        """

        field_data = check_not_tokenized(field_data)

        synsets = disambiguate(field_data)
        synsets = [synset for word, synset in synsets if synset is not None]

        return FeaturesBagField(Counter(synsets))

    def __str__(self):
        return "PyWSDSynsetDocumentFrequency"

    def __repr__(self):
        return "< PyWSDSynsetDocumentFrequency >"
