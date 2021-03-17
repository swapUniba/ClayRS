import os
from typing import List

from orange_cb_recsys.recsys.algorithm import RankingAlgorithm

import pandas as pd

from orange_cb_recsys.utils.const import DEVELOPING, home_path, logger
from orange_cb_recsys.utils.load_content import load_content_instance

from java.nio.file import Paths

from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, BoostQuery
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search.similarities import ClassicSimilarity
from org.apache.lucene.analysis.core import SimpleAnalyzer
from org.apache.lucene.analysis.core import KeywordAnalyzer


class IndexQuery(RankingAlgorithm):
    """
    Class for the search engine recommender
    Args:
        classic_similarity (bool): if you want to use the classic implementation of tfidf in lucene
        positive_threshold (float): ratings bigger than threshold will be considered as positive
    """
    def __init__(self, classic_similarity: bool = True, positive_threshold: float = 0):
        super().__init__(None, None)
        self.__classic_similarity: bool = classic_similarity
        self.__positive_threshold: float = positive_threshold

    def __recs_query(self, positive_rated_document_list: List, scores: List, recs_number: int,
                     items_directory: str, candidate_list: List) -> pd.DataFrame:
        """
        Builds a query using the contents that the user liked. The terms relative to the contents that
        the user liked are boosted by the rating he/she gave. A filter clause is added to the query to
        consider only candidate items
        Args:
            positive_rated_document_list (list): list of contents that the user liked
            scores (list): ratings given by the user
            recs_number (int): how many items must be recommended. Only the number can be specified, not
                a specific item for which compute the prediction
            items_directory (str): directory where the items are stored
            candidate_list (list): list of the items that can be recommended, if None
                all unrated items will be used

        Returns:
            score_frame (pd.DataFrame): dataFrame containing the recommendations for the user
        """
        BooleanQuery.setMaxClauseCount(2000000)
        searcher = IndexSearcher(DirectoryReader.open(SimpleFSDirectory(Paths.get(items_directory))))
        if self.__classic_similarity:
            searcher.setSimilarity(ClassicSimilarity())

        # Obtains the list of fields that each document has in positive_rated_document_list by taking
        # the searching fields in the first document of the list (every document has the same fields),
        # initializes the parser for each of these fields and stores them in a dictionary
        # with the name of the field as key
        field_list = searcher.doc(positive_rated_document_list[0]).getFields()
        field_parsers = {}
        analyzer = SimpleAnalyzer()
        for field in field_list:
            if field.name() == 'content_id':
                continue
            field_parsers[field.name()] = QueryParser(field.name(), analyzer)

        # Initializes user_docs which is a dictionary that has the document as key and
        # another dictionary as value. The dictionary value has the name of the field as key
        # and its contents as value. By doing so we obtain the data of the fields while
        # also storing information regarding the field and the document where it was
        user_docs = {}
        for doc in positive_rated_document_list:
            user_docs[doc] = dict()
            field_list = searcher.doc(doc).getFields()
            for field in field_list:
                if field.name() == 'content_id':
                    continue
                user_docs[doc][field.name()] = field.stringValue()

        logger.info("Building query")

        # Initializes the query through query builder by adding field query.
        # For each field of each document one field_query, containing
        # the data of the field, is created and added to the builder.
        # Also each field query is boosted by the score given by the user to the document
        query_builder = BooleanQuery.Builder()
        for doc, score in zip(user_docs.keys(), scores):
            for field_name in user_docs[doc].keys():
                if field_name == 'content_id':
                    continue
                field_parsers[field_name].setDefaultOperator(QueryParser.Operator.OR)

                field_query = field_parsers[field_name].escape(user_docs[doc][field_name])
                field_query = field_parsers[field_name].parse(field_query)
                field_query = BoostQuery(field_query, score)
                query_builder.add(field_query, BooleanClause.Occur.SHOULD)

        # The requirement of retrieved documents to be in a candidate list (if passed) is added
        # by adding a query with a must clause to the builder
        if candidate_list is not None:
            id_query_string = ' OR '.join("content_id:\"" + content_id + "\"" for content_id in candidate_list)
            id_query = QueryParser("testo_libero", KeywordAnalyzer()).parse(id_query_string)
            query_builder.add(id_query, BooleanClause.Occur.MUST)

        # Retrieves documents
        query = query_builder.build()
        docs_to_search = len(user_docs.keys()) + recs_number
        scoreDocs = searcher.search(query, docs_to_search).scoreDocs

        logger.info("Building score frame to return")

        # Builds the recommendation frame. Also, if one of the retrieved documents was rated positively
        # by the user, it isn't added to the recommendation list
        recorded_items = 0
        columns = ['to_id', 'rating']
        score_frame = pd.DataFrame(columns=columns)
        for scoreDoc in scoreDocs:
            if recorded_items >= recs_number:
                break
            if scoreDoc.doc not in positive_rated_document_list:
                doc = searcher.doc(scoreDoc.doc)
                item_id = doc.getField("content_id").stringValue()
                recorded_items += 1

                score_frame = pd.concat([score_frame, pd.DataFrame.from_records([(item_id, scoreDoc.score)], columns=columns)])

        return score_frame

    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str, candidate_item_id_list: List = None):
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

        scores = []
        rated_document_list = []
        for item_id, score in zip(ratings.to_id, ratings.score):
            if score > self.__positive_threshold:
                item = load_content_instance(items_directory, item_id)
                rated_document_list.append(item.index_document_id)
                scores.append(score)

        return self.__recs_query(rated_document_list,
                                 scores,
                                 recs_number,
                                 index_path,
                                 candidate_item_id_list)
