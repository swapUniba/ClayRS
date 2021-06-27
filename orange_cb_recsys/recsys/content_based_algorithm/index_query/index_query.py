from typing import List
import re

import pandas as pd

from orange_cb_recsys.content_analyzer.memory_interfaces import SearchIndex
from orange_cb_recsys.recsys.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from orange_cb_recsys.utils.const import logger


class IndexQuery(ContentBasedAlgorithm):
    """
    Class for the search engine recommender
    Args:
       item_field (dict): dict where the key is the name of the field
            that contains the content to use, value is the representation(s) that will be
            used for the said item. The value of a field can be a string or a list,
            use a list if you want to use multiple representations for a particular field.
            Check the example above for more.
        classic_similarity (bool): True if you want to use the classic implementation of tfidf in Whoosh,
            False if you want BM25F
        threshold (float): ratings bigger than threshold will be considered as positive
    """

    def __init__(self, item_field: dict, classic_similarity: bool = True, threshold: float = None):
        super().__init__(item_field, threshold)
        self.__string_query: str = None
        self.__scores: list = None
        self.__positive_user_docs: dict = None
        self.__classic_similarity: bool = classic_similarity

    def __get_representations(self, index_representations: dict):
        def find_valid(pattern: str):
            field_index_retrieved = [field_index for field_index in index_representations
                                     if re.match(pattern, field_index)]

            if len(field_index_retrieved) == 0:
                raise KeyError("Id {} not found for the field {}".format(id, k))
            elif len(field_index_retrieved) > 1:
                raise ValueError("This shouldn't happen! Duplicate fields?")
            else:
                valid = field_index_retrieved[0]

            return valid

        representations_valid = {}
        for k in self.item_field:
            for id in self.item_field[k]:
                if isinstance(id, str):
                    pattern = "^{}#.+#{}$".format(k, id)
                else:
                    # the id passed it's a int
                    pattern = "^{}#{}.*$".format(k, id)

                valid_key = find_valid(pattern)
                representations_valid[valid_key] = index_representations[valid_key]

        return representations_valid

    def process_rated(self, user_ratings: pd.DataFrame, index_directory: str):
        """
        Function that extracts features from positive rated items ONLY!
        The extracted features will be used to fit the algorithm (build the query).

        Features extracted will be stored in private attributes of the class.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
        """
        threshold = self.threshold
        if threshold is None:
            threshold = self._calc_mean_user_threshold(user_ratings)

        # Initializes positive_user_docs which is a dictionary that has the document_id as key and
        # another dictionary as value. The dictionary value has the name of the field as key
        # and its contents as value. By doing so we obtain the data of the fields while
        # also storing information regarding the field and the document where it was
        scores = []
        positive_user_docs = {}

        ix = SearchIndex(index_directory)
        for item_id, score in zip(user_ratings.to_id, user_ratings.score):
            if score >= threshold:
                # {item_id: {"item": item_dictionary, "score": item_score}}
                item_query = ix.query(item_id, 1, classic_similarity=self.__classic_similarity)
                if len(item_query) != 0:
                    item = item_query.pop(item_id).get('item')
                    scores.append(score)
                    positive_user_docs[item_id] = self.__get_representations(item)

        self.__positive_user_docs = positive_user_docs
        self.__scores = scores

    def fit(self):
        """
        The fit process for the IndexQuery consists in building a query using the features of the items
        that the user liked. The terms relative to these 'positive' items are boosted by the
        rating he/she gave.

        This method uses extracted features of the positive items stored in a private attribute, so
        process_rated() must be called before this method.

        The built query will also be stored in a private attribute.
        """

        # For each field of each document one string (containing the name of the field and the data in it)
        # is created and added to the query.
        # Also each part of the query that refers to a document
        # is boosted by the score given by the user to said document
        string_query = "("
        for doc, score in zip(self.__positive_user_docs.keys(), self.__scores):
            string_query += "("
            for field_name in self.__positive_user_docs[doc]:
                if field_name == 'content_id':
                    continue
                word_list = self.__positive_user_docs[doc][field_name].split()
                string_query += field_name + ":("
                for term in word_list:
                    string_query += term + " "
                string_query += ") "
            string_query += ")^" + str(score) + " "
        string_query += ") "

        self.__string_query = string_query

    def _build_mask_list(self, user_ratings: pd.DataFrame, filter_list: List[str] = None):
        """
        Private function that calculate the mask query and the filter query for whoosh to use:

        - The mask query is needed to ignore items already rated by the user
        - The filter query is needed to predict only items present in the filter_list

        If in the filter list there are items already rated by the user, those are excluded in the
        mask query so that the prediction for those items can be calculated

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            filter_list (list): list of the items to predict, if None all unrated items will be predicted
        """
        if filter_list is not None:
            masked_list = list(user_ratings.query('to_id not in @filter_list')['to_id'])
        else:
            masked_list = list(user_ratings['to_id'])

        return masked_list

    def predict(self, user_ratings: pd.DataFrame, items_directory: str,
                filter_list: List[str] = None) -> pd.DataFrame:
        """
        IndexQuery is not a Prediction Score Algorithm, so if this method is called,
        a NotPredictionAlg exception is raised

        Raises:
            NotPredictionAlg
        """
        raise NotPredictionAlg("IndexQuery is not a Score Prediction Algorithm!")

    def rank(self, user_ratings: pd.DataFrame, index_directory: str, recs_number: int = None,
             filter_list: List[str] = None) -> pd.DataFrame:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All items will be ranked.

        One can specify which items must be ranked with the filter_list parameter,
        in this case ONLY items in the filter_list will be used to calculate the rank.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items will be used to calculate the rank.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
            recs_number (int): number of the top items that will be present in the ranking
            filter_list (list): list of the items to rank, if None all unrated items will be used to
                calculate the rank
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the rating predicted, sorted in descending order by the 'rating' column
        """

        mask_list = self._build_mask_list(user_ratings, filter_list)

        ix = SearchIndex(index_directory)
        score_docs = ix.query(self.__string_query, recs_number, mask_list, filter_list, self.__classic_similarity)

        logger.info("Building score frame to return")

        results = {'to_id': [], 'score': []}

        for result in score_docs:

            results['to_id'].append(result)
            results['score'].append(score_docs[result]['score'])

        return pd.DataFrame(results)
