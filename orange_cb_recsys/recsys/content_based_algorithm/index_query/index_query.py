from typing import List
import re

import pandas as pd

from orange_cb_recsys.content_analyzer.ratings_manager.ratings import Interaction
from orange_cb_recsys.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from orange_cb_recsys.recsys.content_based_algorithm.contents_loader import LoadedContentsIndex
from orange_cb_recsys.recsys.content_based_algorithm.exceptions import NotPredictionAlg, NoRatedItems, \
    OnlyNegativeItems, EmptyUserRatings


class IndexQuery(ContentBasedAlgorithm):
    """
    Class for the search engine recommender using an index.
    It firsts builds a query using the representation(s) specified of the positive items, then uses the mentioned query
    to do an actual search inside the index: every items will have a score of "closeness" in relation to the
    query, we use this score to rank every item.

    Just be sure to use textual representation(s) to build a significant query and to make a significant search!

        USAGE:
        > # Interested in only a field representation, classic tfidf similarity,
        > # threshold = 3 (Every item with rating >= 3 will be considered as positive)
        > alg = IndexQuery({"Plot": 0}, threshold=3)

        > # Interested in multiple field representations of the items, BM25 similarity,
        > # threshold = 3 (Every item with rating >= 3 will be considered as positive)
        > alg = IndexQuery(
        >                  item_field={"Plot": [0, "original_text"],
        >                              "Genre": [0, 1],
        >                              "Director": "preprocessed_text"},
        >                  classic_similarity=False,
        >                  threshold=3)

        > # After instantiating the IndexQuery algorithm, pass it in the initialization of
        > # a CBRS and the use its method to calculate ranking for single user or multiple users:
        > cbrs = ContentBasedRS(algorithm=alg, ...)
        > cbrs.fit_rank(...)
        > ...
        > # Check the corresponding method documentation for more

    Args:
        item_field (dict): dict where the key is the name of the field
            that contains the content to use, value is the representation(s) id(s) that will be
            used for the said item, just BE SURE to use textual representation(s). The value of a field can be a string
            or a list, use a list if you want to use multiple representations for a particular field.
        classic_similarity (bool): True if you want to use the classic implementation of tfidf in Whoosh,
            False if you want BM25F
        threshold (float): Threshold for the ratings. If the rating is greater than the threshold, it will be considered
            as positive
    """
    __slots__ = ('_string_query', '_scores', '_positive_user_docs', '_classic_similarity')

    def __init__(self, item_field: dict, classic_similarity: bool = True, threshold: float = None):
        super().__init__(item_field, threshold)
        self._string_query: str = None
        self._scores: list = None
        self._positive_user_docs: dict = None
        self._classic_similarity: bool = classic_similarity

    def __get_representations(self, index_representations: dict):
        """
        Private method which extracts representation(s) chosen from all representations codified for the items
        extracted from the index

        Args:
            index_representations (dict): representations for an item extracted from the index
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

    def process_rated(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsIndex):
        """
        Function that extracts features from positive rated items ONLY!
        The extracted features will be used to fit the algorithm (build the query).

        Features extracted will be stored in private attributes of the class.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            index_directory (str): path of the index folder
        """
        items_scores_dict = {interaction.item_id: interaction.score for interaction in user_ratings}

        threshold = self.threshold
        if threshold is None:
            threshold = self._calc_mean_user_threshold(user_ratings)

        # Initializes positive_user_docs which is a dictionary that has the document_id as key and
        # another dictionary as value. The dictionary value has the name of the field as key
        # and its contents as value. By doing so we obtain the data of the fields while
        # also storing information regarding the field and the document where it was
        scores = []
        positive_user_docs = {}

        ix = available_loaded_items.get_contents_interface()

        for item_id, score in zip(items_scores_dict.keys(), items_scores_dict.values()):
            if score >= threshold:
                # {item_id: {"item": item_dictionary, "score": item_score}}
                item_query = ix.query(item_id, 1, classic_similarity=self._classic_similarity)
                if len(item_query) != 0:
                    item = item_query.pop(item_id).get('item')
                    scores.append(score)
                    positive_user_docs[item_id] = self.__get_representations(item)

        if len(user_ratings) == 0:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_id = user_ratings[0].user_id
        if len(positive_user_docs) == 0:
            raise OnlyNegativeItems(f"User {user_id} - There are no rated items available locally or there are only "
                                    f"negative items available locally!")

        self._positive_user_docs = positive_user_docs
        self._scores = scores

    def fit(self):
        """
        The fit process for the IndexQuery consists in building a query using the features of the positive items ONLY
        (items that the user liked). The terms relative to these 'positive' items are boosted by the
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
        for doc, score in zip(self._positive_user_docs.keys(), self._scores):
            string_query += "("
            for field_name in self._positive_user_docs[doc]:
                if field_name == 'content_id':
                    continue
                word_list = self._positive_user_docs[doc][field_name].split()
                string_query += field_name + ":("
                for term in word_list:
                    string_query += term + " "
                string_query += ") "
            string_query += ")^" + str(score) + " "
        string_query += ") "

        self._string_query = string_query

    def _build_mask_list(self, user_seen_items: set, filter_list: List[str] = None):
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
        masked_list = list(user_seen_items)
        if filter_list is not None:
            masked_list = [item for item in user_seen_items if item not in filter_list]

        return masked_list

    def predict(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsIndex,
                filter_list: List[str] = None) -> List[Interaction]:
        """
        IndexQuery is not a Prediction Score Algorithm, so if this method is called,
        a NotPredictionAlg exception is raised

        Raises:
            NotPredictionAlg
        """
        raise NotPredictionAlg("IndexQuery is not a Score Prediction Algorithm!")

    def rank(self, user_ratings: List[Interaction], available_loaded_items: LoadedContentsIndex,
             recs_number: int = None, filter_list: List[str] = None) -> List[Interaction]:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All unrated items will be ranked (or only items in the filter list, if specified).

        One can specify which items must be ranked with the filter_list parameter,
        in this case ONLY items in the filter_list parameter will be ranked.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items will be ranked.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            index_directory (str): path of the index folder
            recs_number (int): number of the top items that will be present in the ranking, if None
                all unrated items will be ranked
            filter_list (list): list of the items to rank, if None all unrated items will be ranked
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the rating predicted, sorted in descending order by the 'rating' column
        """
        try:
            user_id = user_ratings[0].user_id
        except IndexError:
            raise EmptyUserRatings("The user selected doesn't have any ratings!")

        user_seen_items = set([interaction.item_id for interaction in user_ratings])

        mask_list = self._build_mask_list(user_seen_items, filter_list)

        ix = available_loaded_items.get_contents_interface()
        score_docs = ix.query(self._string_query, recs_number, mask_list, filter_list, self._classic_similarity)

        # we construct the output data
        rank_interaction_list = [Interaction(user_id, item_id, score_docs[item_id]['score'])
                                 for item_id in score_docs]

        return rank_interaction_list
