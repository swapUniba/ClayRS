from abc import ABC, abstractmethod
import math


class TfIdf(ABC):

    @abstractmethod
    def compute_tf_idf(self, tf: float, df: float, doc_num: float) -> float:
        """
        Compute tf_idf

        Args:
            tf: term frequency
            df: document frequency
            doc_num: total number of documents in the index

        Returns:
            (float) tf-idf for the given tf and idf values
        """


class TfIdfClassic(TfIdf):

    def compute_tf_idf(self, tf: float, df: float, doc_num: float) -> float:
        return (1 + math.log10(tf)) * math.log10(doc_num/df)

