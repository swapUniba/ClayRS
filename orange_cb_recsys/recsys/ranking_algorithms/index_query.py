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
    """
    def __init__(self, classic_similarity: bool = True, positive_threshold: float = 0):
        super().__init__(None, None)
        self.__classic_similarity: bool = classic_similarity
        self.__positive_threshold: float = positive_threshold

    def __recs_query(self, positive_rated_document_list, scores, recs_number, items_directory,
                     candidate_list: List) -> pd.DataFrame:
        """
        Builds a query using the contents that the user liked. The terms relative to the contents that
        the user liked are boosted by the rating he/she gave. A filter clause is added to the query to
        consider only candidate items
        Args:
            positive_rated_document_list: List of contents that the user liked
            scores: Ratings given by the user
            recs_number: How many items must be recommended. You can only specify the number, not
            a specific item for which compute the prediction
            items_directory: Directory where the items are stored

        Returns:
            score_frame (pd.DataFrame): DataFrame containing the recommendations for the user
        """
        BooleanQuery.setMaxClauseCount(2000000)
        searcher = IndexSearcher(DirectoryReader.open(SimpleFSDirectory(Paths.get(items_directory))))
        if self.__classic_similarity:
            searcher.setSimilarity(ClassicSimilarity())

        field_list = searcher.doc(positive_rated_document_list[0]).getFields()
        user_fields = {}
        field_parsers = {}
        analyzer = SimpleAnalyzer()
        for field in field_list:
            if field.name() == 'content_id':
                continue
            user_fields[field.name()] = field.stringValue()
            field_parsers[field.name()] = QueryParser(field.name(), analyzer)

        positive_rated_document_list.remove(positive_rated_document_list[0])

        for _ in positive_rated_document_list:
            for field in field_list:
                if field.name() == 'content_id':
                    continue
                user_fields[field.name()] += field.stringValue()

        logger.info("Building query")

        query_builder = BooleanQuery.Builder()
        for score in scores:
            for field_name in user_fields.keys():
                if field_name == 'content_id':
                    continue
                field_parsers[field_name].setDefaultOperator(QueryParser.Operator.OR)

                field_query = field_parsers[field_name].escape(user_fields[field_name])
                field_query = field_parsers[field_name].parse(field_query)
                field_query = BoostQuery(field_query, score)
                query_builder.add(field_query, BooleanClause.Occur.SHOULD)

        if candidate_list is not None:
            id_query_string = ' OR '.join("content_id:\"" + content_id + "\"" for content_id in candidate_list)
            id_query = QueryParser("testo_libero", KeywordAnalyzer()).parse(id_query_string)
            query_builder.add(id_query, BooleanClause.Occur.MUST)

        query = query_builder.build()
        docs_to_search = len(positive_rated_document_list) + recs_number
        scoreDocs = searcher.search(query, docs_to_search).scoreDocs

        logger.info("Building score frame to return")

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

    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number, items_directory: str, candidate_item_id_list: List = None):
        """
        Finds the documents that the user liked and then calls __recs_query to execute the prediction
        Args:
            candidate_item_id_list: list of the items that can be recommended, if None
            all unrated items will be used
            user_id: user for which recommendations will be computed
            recs_number (list[Content]): How long the ranking will be
            ratings (pd.DataFrame): ratings of the user with id equal to user_id
            items_directory (str): Name of the directory where the items are stored.
        Returns:
            (pd.DataFrame)
        """
        index_path = os.path.join(items_directory, 'search_index')
        if not DEVELOPING:
            index_path = os.path.join(home_path, items_directory, 'search_index')

        scores = []
        rated_document_list = []
        for item_id, score in zip(ratings.to_id, ratings.score):
            item = load_content_instance(items_directory, item_id)

            if score > self.__positive_threshold:
                rated_document_list.append(item.index_document_id)
                scores.append(score)

        return self.__recs_query(rated_document_list,
                                 scores,
                                 recs_number,
                                 index_path,
                                 candidate_item_id_list)
