from __future__ import annotations
import re
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from clayrs.content_analyzer.information_processor.information_processor_abstract import InformationProcessor
    from clayrs.content_analyzer.raw_information_source import RawInformationSource

from clayrs.content_analyzer.field_content_production_techniques.field_content_production_technique import \
    SynsetDocumentFrequency
from clayrs.content_analyzer.utils.check_tokenization import check_not_tokenized
from clayrs.utils.context_managers import get_progbar


class PyWSDSynsetDocumentFrequency(SynsetDocumentFrequency):
    """
    Class that produces a sparse vector for each content representing the document frequency of each synset found inside
    the document. The synsets are computed thanks to ***PyWSD*** library.

    Consider this textual representation:
    ```
    content1: "After being trapped in a jungle board game for 26 years"
    content2: "After considering jungle County, it was trapped in a jungle"
    ```

    This technique will produce the following sparse vectors:

    ```
    # vocabulary of the features
    vocabulary = {'trap.v.04': 4, 'jungle.n.03': 2, 'board.n.09': 0,
                  'plot.n.01': 3, 'twenty-six.s.01': 5,
                  'year.n.03': 7, 'view.v.02': 6, 'county.n.02': 1}

    content1:
        (0, 4)	1
        (0, 2)	1
        (0, 0)	1
        (0, 3)	1
        (0, 5)	1
        (0, 7)	1

    content2:
        (0, 4)	1
        (0, 2)	2
        (0, 6)	1
        (0, 1)	1
    ```

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
        return "PyWSDSynsetDocumentFrequency()"
