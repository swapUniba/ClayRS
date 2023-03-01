from __future__ import annotations
from collections import defaultdict
from typing import List, Optional, Set, TYPE_CHECKING
import re
import numpy as np

if TYPE_CHECKING:
    from clayrs.content_analyzer.ratings_manager.ratings import Ratings

from clayrs.recsys.content_based_algorithm.content_based_algorithm import PerUserCBAlgorithm
from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsIndex
from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg, OnlyNegativeItems, EmptyUserRatings


class IndexQuery(PerUserCBAlgorithm):
    """
    Class for the search engine recommender using an index.
    It firsts builds a query using the representation(s) specified of the positive items, then uses the mentioned query
    to do an actual search inside the index: every item will have a score of "closeness" in relation to the
    query, we use this score to rank every item.

    Just be sure to use textual representation(s) to build a significant query and to make a significant search!

    Examples:

        * Interested in only a field representation, classic tfidf similarity,
        `threshold` $= 3$ (Every item with rating $>= 3$ will be considered as positive)

        >>> from clayrs import recsys as rs
        >>> alg = rs.IndexQuery({"Plot": 0}, threshold=3)

        * Interested in multiple field representations of the items, BM25 similarity,
        `threshold` $= None$ (Every item with rating $>=$ mean rating of the user will be considered as positive)

        >>> alg = rs.IndexQuery(
        >>>                     item_field={"Plot": [0, "original_text"],
        >>>                                 "Genre": [0, 1],
        >>>                                 "Director": "preprocessed_text"},
        >>>                     classic_similarity=False,
        >>>                     threshold=3)


        !!! info

            After instantiating the IndexQuery algorithm, pass it in the initialization of
            a CBRS and the use its method to calculate ranking for single user or multiple users:

            Examples:

                >>> cbrs = rs.ContentBasedRS(algorithm=alg, ...)
                >>> cbrs.fit_rank(...)
                >>> # ...

    Args:
        item_field: dict where the key is the name of the field
            that contains the content to use, value is the representation(s) id(s) that will be
            used for the said item, just BE SURE to use textual representation(s). The value of a field can be a string
            or a list, use a list if you want to use multiple representations for a particular field.
        classic_similarity: True if you want to use the classic implementation of tfidf in Whoosh,
            False if you want BM25F
        threshold: Threshold for the ratings. If the rating is greater than the threshold, it will be considered
            as positive. If the threshold is not specified, the average score of all items rated by the user is used.
    """
    __slots__ = ('_string_query', '_scores', '_positive_user_docs', '_classic_similarity')

    def __init__(self, item_field: dict, classic_similarity: bool = True, threshold: float = None):
        super().__init__(item_field, threshold)
        self._string_query: Optional[str] = None
        self._scores: Optional[list] = None
        self._positive_user_docs: Optional[dict] = None
        self._classic_similarity: bool = classic_similarity

    def _get_representations(self, index_representations: dict):
        """
        Private method which extracts representation(s) chosen from all representations codified for the items
        extracted from the index

        Args:
            index_representations: representations for an item extracted from the index
        """

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
                # every representation for an item is codified like this: plot#0#tfidf
                if isinstance(id, str):
                    pattern = "^{}#.+#{}$".format(k, id)
                else:
                    # the id passed it's a int
                    pattern = "^{}#{}.*$".format(k, id)

                valid_key = find_valid(pattern)
                representations_valid[valid_key] = index_representations[valid_key]

        return representations_valid

    def _load_available_contents(self, index_path: str, items_to_load: set = None):
        return LoadedContentsIndex(index_path)

    def process_rated(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsIndex):
        """
        Function that extracts features from positive rated items ONLY of a user
        The extracted features will be used to fit the algorithm (build the query).

        Features extracted will be stored in private attributes of the class.

        IF there are no rated items available locally or if there are only positive/negative
        items, an exception is thrown.

        Args:
            user_idx: Mapped integer of the active user (the user for which we must fit the algorithm)
            train_ratings: `Ratings` object which contains the train set of each user
            available_loaded_items: The LoadedContents interface which contains loaded contents

        Raises:
            EmptyUserRatings: Exception raised when the user does not appear in the train set
            OnlyNegativeitems: Exception raised when there are only negative items available locally
                for the user (Items that the user disliked)
        """

        uir_user = train_ratings.get_user_interactions(user_idx)
        rated_items_id = train_ratings.item_map.convert_seq_int2str(uir_user[:, 1].astype(int))

        # a list since there could be duplicate interaction (eg bootstrap partitioning)
        items_scores_dict = defaultdict(list)

        for item_id, score in zip(rated_items_id, uir_user[:, 2]):
            items_scores_dict[item_id].append(score)

        items_scores_dict = dict(sorted(items_scores_dict.items()))  # sort dictionary based on key for reproducibility

        threshold = self.threshold
        if threshold is None:
            threshold = np.nanmean(uir_user[:, 2])

        # Initializes positive_user_docs which is a list that has tuples with document_id as first element and
        # a dictionary as second. The dictionary value has the name of the field as key
        # and its contents as value. By doing so we obtain the data of the fields while
        # also storing information regarding the field and the document where it was
        scores = []
        positive_user_docs = []

        ix = available_loaded_items.get_contents_interface()

        # we extract feature of each item sorted based on its key: IMPORTANT for reproducibility!!
        # we must convert keys (which are strings) to the respective int idx to build the uir
        for (item_id, (item_idx, score_list)) in zip(rated_items_id, items_scores_dict.items()):

            score_assigned = map(float, score_list)

            for score in score_assigned:
                if score >= threshold:
                    # {item_id: {"item": item_dictionary, "score": item_score}}
                    item_query = ix.query(item_id, results_number=1, classic_similarity=self._classic_similarity)
                    if len(item_query) != 0:
                        item = item_query.pop(item_id).get('item')
                        scores.append(score)
                        positive_user_docs.append((item_idx, self._get_representations(item)))

        if len(uir_user[:, 1]) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        if len(positive_user_docs) == 0:
            raise OnlyNegativeItems(f"User {user_idx} - There are no rated items available locally or there are only "
                                    f"negative items available locally!")

        self._positive_user_docs = positive_user_docs
        self._scores = scores

    def fit_single_user(self):
        """
        The fit process for the IndexQuery consists in building a query using the features of the positive items ONLY
        (items that the user liked). The terms relative to these 'positive' items are boosted by the
        rating he/she/it gave.

        This method uses extracted features of the positive items stored in a private attribute, so
        `process_rated()` must be called before this method.

        The built query will also be stored in a private attribute.
        """
        # For each field of each document one string (containing the name of the field and the data in it)
        # is created and added to the query.
        # Also each part of the query that refers to a document
        # is boosted by the score given by the user to said document
        string_query = "("
        for (doc_id, doc_data), score in zip(self._positive_user_docs, self._scores):
            string_query += "("
            for field_name in doc_data:
                if field_name == 'content_id':
                    continue
                word_list = doc_data[field_name].split()
                string_query += field_name + ":("
                for term in word_list:
                    string_query += term + " "
                string_query += ") "
            string_query += ")^" + str(score) + " "
        string_query += ") "

        self._string_query = string_query

    def _build_mask_list(self, user_seen_items: Set[str], filter_list: List[str]):
        """
        Private function that calculate the mask query and the filter query for whoosh to use:

        - The mask query is needed to ignore items already rated by the user
        - The filter query is needed to predict only items present in the filter_list

        If in the filter list there are items already rated by the user, those are excluded in the
        mask query so that the prediction for those items can be calculated

        Args:
            user_seen_items: set of items present in the user profile
            filter_list: list of the items to predict, if None all unrated items will be predicted
        """
        masked_list = [item for item in user_seen_items if item not in filter_list]

        return masked_list

    def predict_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsIndex,
                            filter_list: List[str]) -> np.ndarray:
        """
        IndexQuery is not a Prediction Score Algorithm, so if this method is called,
        a NotPredictionAlg exception is raised

        Raises:
            NotPredictionAlg: exception raised since the IndexQuery algorithm is not a score prediction algorithm
        """
        raise NotPredictionAlg("IndexQuery is not a Score Prediction Algorithm!")

    def rank_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsIndex,
                         recs_number: Optional[int], filter_list: List[str]) -> np.ndarray:
        """
        Rank the top-n recommended items for the active user, where the top-n items to rank are controlled by the
        `recs_number` and `filter_list` parameter:

        * the former one is self-explanatory, the second is a list of items
        represented with their string ids. Must be necessarily strings and not their mapped integer since items are
        serialized following their string representation!

        If `recs_number` is `None`, all ranked items will be returned

        The filter list parameter is usually the result of the `filter_single()` method of a `Methodology` object

        Args:
            user_idx: Mapped integer of the active user
            train_ratings: `Ratings` object which contains the train set of each user
            available_loaded_items: The LoadedContents interface which contains loaded contents
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            filter_list: list of the items to rank. Should contain string item ids

        Returns:
            uir matrix for a single user containing user and item idxs (integer representation) with the ranked score
                as third dimension sorted in a decreasing order
        """
        uir_user = train_ratings.get_user_interactions(user_idx)
        if len(uir_user) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_seen_items = train_ratings.item_map.convert_seq_int2str(uir_user[:, 1].astype(int))
        mask_list = self._build_mask_list(user_seen_items, filter_list)

        ix = available_loaded_items.get_contents_interface()
        score_docs = ix.query(self._string_query, recs_number, mask_list, filter_list, self._classic_similarity)

        # we must convert keys (which are strings) to the respective int idx to build the uir
        score_list_idxs = train_ratings.item_map.convert_seq_str2int(list(score_docs.keys()))

        # we construct the output data
        uir_rank = np.array([[user_idx, item_idx, score_docs[item_id]['score']]
                             for item_idx, item_id in zip(score_list_idxs, score_docs)])

        return uir_rank

    def __str__(self):
        return "IndexQuery"

    def __repr__(self):
        return f'IndexQuery(item_field={self.item_field}, classic_similarity={self._classic_similarity}, ' \
               f'threshold={self.threshold})'
