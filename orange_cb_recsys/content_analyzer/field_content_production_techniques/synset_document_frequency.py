from sklearn.feature_extraction.text import CountVectorizer

from orange_cb_recsys.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    SynsetDocumentFrequency
from orange_cb_recsys.content_analyzer.information_processor.information_processor import InformationProcessor
from orange_cb_recsys.content_analyzer.raw_information_source import RawInformationSource
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized
from typing import List

import re

from orange_cb_recsys.utils.const import get_progbar


class PyWSDSynsetDocumentFrequency(SynsetDocumentFrequency):
    """
    Pywsd word sense disambiguation
    """
    def __init__(self):
        # The import is here since pywsd has a long warm up phase that should affect the computation
        # only when effectively instantiated
        from pywsd import disambiguate

        self.disambiguate = disambiguate
        super().__init__()

    def dataset_refactor(self, information_source: RawInformationSource, field_name: str,
                         preprocessor_list: List[InformationProcessor]):

        all_synsets = []
        with get_progbar(information_source) as pbar:
            pbar.set_description("Computing synset frequency with wordnet")
            for raw_content in pbar:
                processed_field_data = self.process_data(raw_content[field_name], preprocessor_list)

                processed_field_data = check_not_tokenized(processed_field_data)

                synset_list = ' '.join([synset.name()
                                        for _, synset in self.disambiguate(processed_field_data)
                                        if synset is not None])
                all_synsets.append(synset_list)

        # tokenizer based on whitespaces since one document is represented as 'mysynset.id.01 mysynset.id.02 ...'
        def split_tok(text):
            return re.split("\\s+", text)

        cv = CountVectorizer(tokenizer=split_tok)
        res = cv.fit_transform(all_synsets)

        self._synset_matrix = res
        self._synset_names = cv.get_feature_names_out()

        return self._synset_matrix.shape[0]

    def __str__(self):
        return "PyWSDSynsetDocumentFrequency"

    def __repr__(self):
        return "< PyWSDSynsetDocumentFrequency >"
