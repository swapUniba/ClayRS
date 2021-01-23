from typing import List

from sklearn.feature_extraction import DictVectorizer
from scipy import sparse
from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.algorithm import RankingAlgorithm
from orange_cb_recsys.recsys.ranking_algorithms.similarities import Similarity, DenseVector, SparseVector
from orange_cb_recsys.content_analyzer.content_representation.content_field import EmbeddingField, FeaturesBagField
import pandas as pd
import numpy as np

from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.load_content import get_unrated_items, get_rated_items, load_content_instance


class CentroidVector(RankingAlgorithm):
    """
    Class that implements a centroid-like recommender. It first gets the centroid of the items that the user liked.
    Then computes the similarity between the centroid and the item of which predict the score.
    Args:
        item_field: Name of the field that contains the content to use
        field_representation: Id of the field_representation content of which compute the centroid
        similarity (Similarity): Kind of similarity to use
        threshold (int): Threshold for the ratings. If the rating is greater than the threshold, it will be considered
            as positive
    """

    def __init__(self, item_field: str, field_representation: str, similarity: Similarity, threshold: int = 0):
        super().__init__(item_field, field_representation)
        self.__similarity = similarity
        self.__threshold = threshold

    def __get_centroid_with_vectorizer(self, ratings: pd.DataFrame, rated_items, unrated_items):
        """
        1) For each rated item, checks if its rating is bigger than threshold. If false, skips
        to the next item, if True add the item embedding array in a dictionary list taht will be
        transformed in a scipy  csr_matrix (sparse) using sklearn DictVectorizer
        2) Computes the centroid of the obtained sparse matrix

        Args:
            ratings (pd.DataFrame): DataFrame containing the ratings.

        Returns:
            centroid (sparse.csr_matrix): Sparse matrix that represents the centroid vector of the
                given item representations
        """

        dv = DictVectorizer(sparse=True)

        positive_rated_items = [
            item.get_field(self.item_field).get_representation(self.item_field_representation).value
            for item in rated_items
            if float(ratings[ratings['to_id'] == item.content_id].score) >= self.__threshold]
        unr =[]
        for item in unrated_items:
          item: Content = item
          if item is not None:
            unr.append(item.get_field(self.item_field).get_representation(self.item_field_representation).value)

        dicts = positive_rated_items + unr

        matrix = dv.fit_transform(dicts)
        return sparse.csr_matrix(matrix.mean(axis=0).getA()), matrix[
                                                              len(rated_items):len(rated_items) + len(unrated_items)]

    def __get_centroid_without_vectorizer(self, ratings: pd.DataFrame, rated_items) -> np.ndarray:
        """
        1) For each rated item, checks if its rating is bigger than threshold. If false, skips
        to the next item, if True add the item embedding array in a matrix
        2) Computes the centroid of the obtained matrix

        Args:
            ratings (pd.DataFrame): DataFrame containing the ratings.

        Returns:
            centroid (np.array): numpy array that represents the centroid vector of the
                given item representations
        """

        arrays = []
        for item in rated_items:
            representation = item.get_field(self.item_field).get_representation(
                self.item_field_representation)
            if float(ratings[ratings['to_id'] == item.content_id].score) >= self.__threshold:
                arrays.append(representation.value)
        return np.array(arrays).mean(axis=0)

    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None) -> pd.DataFrame:
        """
        Checks:
        1) Checks if the representation corresponding to field_representation exists
        2) Checks if the field representation is a document embedding (whose shape equals 1)

        Example: item_field == "Plot" and field_representation == "1", the function will check if the "01"
        representation of each "Plot" field is a document embedding or a tf-idf words bag, and then use the embedding
        or the frequency vector for algorithm computation.

        Computes the centroid of the positive rated items representations

        For each candidate item:
        1) Takes the embedding arrays
        2) Determines the similarity between the centroid and the field_representation of the item_field in candidate item.

        Args:
            candidate_item_id_list: list of the items that can be recommended, if None
                all unrated items will be used
            user_id: user for which recommendations will be computed
            recs_number (list[Content]): How long the ranking will be
            ratings (pd.DataFrame): ratings of the user with id equal to user_id
            items_directory (str): Name of the directory where the items are stored.

        Returns:
             scores (pd.DataFrame): DataFrame whose columns are the ids of the items (to_id), and the similarities between the
                  items and the centroid (rating)
        """

        # try:
        logger.info("Retrieving candidate items")
        if candidate_item_id_list is None:
            unrated_items = get_unrated_items(items_directory, ratings)
        else:
            unrated_items = [load_content_instance(items_directory, item_id) for item_id in candidate_item_id_list]

        logger.info("Retrieving rated items")
        rated_items = get_rated_items(items_directory, ratings)
        if len(rated_items) == 0:
          columns = ["to_id", "rating"]
          scores = pd.DataFrame(columns=columns)
          return scores
        first_item = rated_items[0]
        need_vectorizer = False
        if self.item_field not in first_item.field_dict:
            raise ValueError("The field name specified could not be found!")
        else:
            try:
                representation = first_item.get_field(self.item_field).get_representation(
                    self.item_field_representation)
            except KeyError:
                raise ValueError("The given representation id wasn't found for the specified field")

            if not isinstance(representation, EmbeddingField) and not isinstance(representation, FeaturesBagField):
                raise ValueError("The given representation must be an embedding or a tf-idf vector")

            if isinstance(representation, EmbeddingField):
                if len(representation.value.shape) != 1:
                    raise ValueError("The specified representation is not a document embedding, so the centroid"
                                     " can not be calculated")

            if isinstance(representation, FeaturesBagField):
                need_vectorizer = True

        columns = ["to_id", "rating"]
        scores = pd.DataFrame(columns=columns)

        if not need_vectorizer:
            logger.info("Computing centroid")
            centroid = self.__get_centroid_without_vectorizer(ratings, rated_items)
            logger.info("Computing similarities")

            for item in unrated_items:
                item_id = item.content_id
                item_field_representation = item.get_field(self.item_field).get_representation(
                    self.item_field_representation).value
                logger.info("Computing similarity with %s" % item_id)
                similarity = self.__similarity.perform(DenseVector(centroid), DenseVector(item_field_representation))
                scores = pd.concat([scores, pd.DataFrame.from_records([(item_id, similarity)], columns=columns)],
                                   ignore_index=True)
        else:
            logger.info("Computing centroid")
            centroid, unrated_matrix = self.__get_centroid_with_vectorizer(ratings, rated_items, unrated_items)

            logger.info("Computing similarities")

            a = []
            for x in unrated_items:
              if x is not None:
                a.append(x)
            unrated_items = a

            for item, item_array in zip(unrated_items, unrated_matrix):
                item_id = item.content_id
                logger.info("Computing similarity with %s" % item_id)
                similarity = self.__similarity.perform(SparseVector(centroid), SparseVector(item_array))
                scores = pd.concat([scores, pd.DataFrame.from_records([(item_id, similarity)], columns=columns)],
                                   ignore_index=True)

        scores = scores.sort_values(['rating'], ascending=False).reset_index(drop=True)
        scores = scores[:recs_number]

        return scores
        # except ValueError as v:
        #     raise ValueError
        #     #print(str(v))
