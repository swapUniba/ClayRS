import os

from orange_cb_recsys.recsys.algorithm import RankingAlgorithm

import pandas as pd

from orange_cb_recsys.utils.const import DEVELOPING, home_path, logger
from orange_cb_recsys.utils.load_content import load_content_instance, remove_not_existent_items

from whoosh.index import open_dir
from whoosh.query import Term, Or
from whoosh.qparser import QueryParser
from whoosh import scoring, qparser


class IndexQuery(RankingAlgorithm):
    """
    Class for the search engine recommender
    Args:
        classic_similarity (bool): True if you want to use the classic implementation of tfidf in Whoosh,
            False if you want BM25F
        positive_threshold (float): ratings bigger than threshold will be considered as positive
    """
    def __init__(self, classic_similarity: bool = True, positive_threshold: float = 0):
        super().__init__(None, None)
        self.__classic_similarity: bool = classic_similarity
        self.__positive_threshold: float = positive_threshold

    def __recs_query(self, positive_rated_document_list: list, rated_document_list: list,
                     scores: list, recs_number: int, items_directory: str, candidate_list: list) -> pd.DataFrame:
        """
        Builds a query using the contents that the user liked. The terms relative to the contents that
        the user liked are boosted by the rating he/she gave. A filter clause is added to the query to
        consider only candidate items
        Args:
            positive_rated_document_list (list): list of contents that the user liked
            rated_document_list (list): list of all the contents that the user rated
            scores (list): ratings given by the user
            recs_number (int): how many items must be recommended. Only the number can be specified, not
                a specific item for which compute the prediction
            items_directory (str): directory where the items are stored
            candidate_list (list): list of the items that can be recommended, if None
                all unrated items will be used

        Returns:
            score_frame (pd.DataFrame): dataFrame containing the recommendations for the user
        """
        ix = open_dir(items_directory)
        with ix.searcher(weighting=scoring.TF_IDF if self.__classic_similarity else scoring.BM25F) as searcher:

            # Initializes user_docs which is a dictionary that has the document as key and
            # another dictionary as value. The dictionary value has the name of the field as key
            # and its contents as value. By doing so we obtain the data of the fields while
            # also storing information regarding the field and the document where it was
            field_list = None
            user_docs = {}
            for doc in positive_rated_document_list:
                user_docs[doc] = dict()
                field_list = searcher.stored_fields(doc)
                for field_name in field_list:
                    if field_name == 'content_id':
                        continue
                    user_docs[doc][field_name] = field_list[field_name]

            logger.info("Building query")

            # For each field of each document one string (containing the name of the field and the data in it)
            # is created and added to the query.
            # Also each part of the query that refers to a document
            # is boosted by the score given by the user to said document
            string_query = "("
            for doc, score in zip(user_docs.keys(), scores):
                string_query += "("
                for field_name in field_list:
                    if field_name == 'content_id':
                        continue
                    word_list = user_docs[doc][field_name].split()
                    string_query += field_name + ":("
                    for term in word_list:
                        string_query += term + " "
                    string_query += ") "
                string_query += ")^" + str(score) + " "
            string_query += ") "

            # The requirement of retrieved documents to be in a candidate list (if passed) is added
            # by building a query for the content id of said documents.
            # Also the query containing all the content ids for the documents that the user rated
            # is created.
            # Both these queries will be used by the index searcher
            candidate_query_list = None
            rated_query_list = []

            for document in rated_document_list:
                rated_query_list.append(Term("content_id", document))
            rated_query_list = Or(rated_query_list)

            if candidate_list is not None:
                candidate_query_list = []
                for candidate in candidate_list:
                    candidate_query_list.append(Term("content_id", candidate))
                candidate_query_list = Or(candidate_query_list)

            # The filter and mask arguments of the index searcher are used respectively
            # to find only candidate documents or to ignore documents rated by the user
            schema = ix.schema
            query = QueryParser("content_id", schema=schema, group=qparser.OrGroup).parse(string_query)
            score_docs = searcher.search(query, limit=recs_number, filter=candidate_query_list, mask=rated_query_list)

            logger.info("Building score frame to return")

            # Builds the recommendation frame. Items in the candidate list or rated by the user
            # were already filtered previously by the index searcher
            columns = ['to_id', 'rating']
            score_frame = pd.DataFrame(columns=columns)
            for result in score_docs:
                item_id = result["content_id"]

                score_frame = pd.concat([
                    score_frame, pd.DataFrame.from_records([(item_id, result.score)], columns=columns)])

        return score_frame

    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: list = None):
        """
        Finds the documents that the user liked by comparing the score given by the user to the item
        against the positive_threshold of the index_query object (if the rating is greater than the threshold,
        the document it refers to is considered liked by the user)
        After that, calls __recs_query to execute the prediction
        Args:
            user_id (str): user for which recommendations will be computed
            ratings (pd.DataFrame): ratings of the user with id equal to user_id
            recs_number (int): how long the ranking will be
            items_directory (str): name of the directory where the items are stored
            candidate_item_id_list (list): list of the items that can be recommended, if None
                all unrated items will be used
        Returns:
            (pd.DataFrame) dataframe that for each row has a suggested item id and a rating of
                said item. This rating represents how much the item matches the query used for
                retrieving the recommendation list
        EXAMPLES:
            Find a recommendation list with two items for a user:
                predict('A000', ratings, 2, '../../example')
            Find a recommendation list with one item for a user considering a candidate list containing two items:
                predict('A000', ratings, 1, '../../example', ['tt0114885', 'tt0114388'])
            Ratings is a variable containing a dataframe with the user ratings
            Ratings dataframe columns example: "from_id", "to_id", "original_rating", "score", "timestamp"
        """
        index_path = os.path.join(items_directory, 'search_index')
        if not DEVELOPING:
            index_path = os.path.join(home_path, items_directory, 'search_index')

        valid_ratings = remove_not_existent_items(ratings, items_directory)
        scores = []
        positive_rated_document_list = []
        for item_id, score in zip(valid_ratings.to_id, valid_ratings.score):
            if score > self.__positive_threshold:
                item = load_content_instance(items_directory, item_id)
                positive_rated_document_list.append(item.index_document_id)
                scores.append(score)

        return self.__recs_query(positive_rated_document_list,
                                 valid_ratings.to_id,
                                 scores,
                                 recs_number,
                                 index_path,
                                 candidate_item_id_list)
