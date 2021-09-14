import abc
from typing import List
import pandas as pd
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.validation import check_is_fitted
import numpy as np

from orange_cb_recsys.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique
from orange_cb_recsys.recsys.algorithm import Algorithm

from orange_cb_recsys.content_analyzer.content_representation.content import Content


class ContentBasedAlgorithm(Algorithm):
    """
    Abstract class for the content-based algorithms

    Every Content Based Algorithm base its prediction (let it be score prediction or ranking) on representations
    serialized by the Content Analyzer. It can be used a single representation or multiple ones for a single field or
    multiple ones

    Args:
    item_field (dict): dict where the key is the name of the field
        that contains the content to use, value is the representation(s) id(s) that will be
        used for the said item. The value of a field can be a string or a list,
        use a list if you want to use multiple representations for a particular field.
    threshold (float): Threshold for the ratings. Some algorithms use this threshold in order to
        separate positive items from the negative ones, others may use only ratings that are >= than this
        threshold. See the documentation of the algorithm used for more
    """

    def __init__(self, item_field: dict, threshold: float):
        self.item_field: dict = self._bracket_representation(item_field)
        self.threshold: float = threshold
        self.__transformer = None

    def _set_transformer(self):
        """
        Private method that set the transformer later used in order to fuse multiple representations
        """
        self.__transformer = DictVectorizer(sparse=True, sort=False)

    @staticmethod
    def _bracket_representation(item_field: dict):
        """
        Private method that brackets every representation in case the user passed a string
        instead of a list.

        EXAMPLE::
            > item_field = {'Plot': 0, 'Genre': ['tfidf', 1]}
            > print(_bracket_representation(item_field))

            > {'Plot': [0], 'Genre': ['tfidf', 1]}

        Args:
            item_field (dict): dict that may contain values that need to be bracketed

        Returns:
            The item_field passed with all values inside a list
        """
        for field in item_field:
            if not isinstance(item_field[field], list):
                item_field[field] = [item_field[field]]

        return item_field

    @staticmethod
    def _calc_mean_user_threshold(user_ratings: pd.DataFrame):
        """
        Private method which simply calculates the average rating by the user given its ratings
        """
        return user_ratings['score'].mean()

    def extract_features_item(self, item: Content):
        """
        Function that extracts the feature of a loaded item using the item_field parameter passed in the
        constructor.

        It extracts only the chosen representations of the chosen fields in the item loaded
        EXAMPLE:

            with item_field = {'Plot': [0], 'Genre': ['tfidf', 1]}, the function will extracts
            only the representation with '0' as internal id for the field 'Plot' and two representations
            for the field 'Genre': one with 'tfidf' as external id and the other with 1 as internal id

        Args:
            item (Content): item loaded of which we need to extract its feature

        Returns:
            A list containing all representations extracted for the item
        """
        item_bag_list = []
        if item is not None:
            for field in self.item_field:
                field_representations = self.item_field[field]

                for representation in field_representations:
                    item_bag_list.append(
                        item.get_field_representation(field, representation).value
                    )

        return item_bag_list

    def fuse_representations(self, X: list, embedding_combiner: CombiningTechnique):
        """
        Method which transforms the X passed vectorizing if X contains dicts and merging
        multiple representations in a single one for every item in X.
        So if X = [
                    [dict, np.array, np.array]
                        ...
                    [dict, np.array, np.array]
                ]
        where every sublist contains multiple representation for a single item,
        the function returns:
        X = [
                np.array,
                ...
                np.array
            ]
        Where every row is the fused representation for the item

        In case np.array have different row size, every array will be transformed in a one dimensional one
        using the parameter embedding combiner. Check all the available combining technique to know how rows of
        a np.array can be merged into one

        Args:
            X (list): list that contains representations of the items
            embedding_combiner (CombiningTechnique): combining technique in case there are multiple
                vectors with different row size
        Returns:
            X fused and vectorized
        """
        if self.__transformer is None:
            raise ValueError("Transformer not set! Every CB Algorithm must call the method _set_transformer()"
                             " in its fit() method")

        if any(not isinstance(rep, dict) and not isinstance(rep, np.ndarray) and not isinstance(rep, float) for rep in X[0]):
            raise ValueError("You can only use representations of type: {numeric, embedding, tfidf}")

        # We check if there are dicts as representation in the first element of X,
        # since the representations are the same for all elements in X we can check
        # for dicts only in one element
        need_vectorizer = any(isinstance(rep, dict) for rep in X[0])

        if need_vectorizer:
            # IF the transformer is not fitted then we are training the model
            try:
                check_is_fitted(self.__transformer)
            except NotFittedError:
                X_dicts = [rep for item in X for rep in item if isinstance(rep, dict)]
                self.__transformer.fit(X_dicts)

        # In every case, we transform the input
        X_vectorized_sparse = []
        for sublist in X:
            single_sparse = sparse.csr_matrix((1, 0))
            for item in sublist:
                if need_vectorizer and isinstance(item, dict):
                    vector = self.__transformer.transform(item)
                    single_sparse = sparse.hstack((single_sparse, vector), format='csr')
                elif isinstance(item, np.ndarray):
                    if item.ndim > 1:
                        item = embedding_combiner.combine(item)

                    item_sparse = sparse.csr_matrix(item)
                    single_sparse = sparse.hstack((single_sparse, item_sparse), format='csr')
                else:
                    # it's a float
                    item_sparse = sparse.csr_matrix(item)
                    single_sparse = sparse.hstack((single_sparse, item_sparse), format='csr')

            X_vectorized_sparse.append(single_sparse)

        X_dense = [x.toarray().flatten() for x in X_vectorized_sparse]

        return X_dense

    @abc.abstractmethod
    def process_rated(self, user_ratings: pd.DataFrame, items_directory: str):
        """
        Abstract method that processes rated items for the user.

        Every content-based algorithm processes rated items differently, it may be needed to extract features
        from the rated items and label them, extract features only from the positive ones, etc.

        The rated items processed must be stored into a private attribute of the algorithm, later used
        by the fit() method.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self):
        """
        Abstract method that fits the content-based algorithm.

        Every content based algorithm has a different fit process, it may be needed to fit a classifier,
        to build the centroid of the positive items, to build a query for the index, etc.

        It must be called after the the process_rated() method since it uses private attributes calculated
        by said method to fit the algorithm.

        The fitted object will also be stored in a private attribute.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, user_ratings: pd.DataFrame, items_directory: str,
                filter_list: List[str] = None) -> pd.DataFrame:
        """
        |  Abstract method that predicts the rating which a user would give to items
        |  If the algorithm is not a PredictionScore Algorithm, implement this method like this:

        def predict():
            raise NotPredictionAlg

        One can specify which items must be predicted with the filter_list parameter,
        in this case ONLY items in the filter_list will be predicted.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items will be predicted.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            items_directory (str): path of the directory where the items are stored
            filter_list (list): list of the items to predict, if None all unrated items will be score predicted
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the score predicted
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, user_ratings: pd.DataFrame, items_directory: str, recs_number: int = None,
             filter_list: List[str] = None) -> pd.DataFrame:
        """
        |  Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        |  all items will be ranked.
        |  If the algorithm is not a Ranking Algorithm, implement this method like this:

        def rank():
            raise NotRankingAlg

        One can specify which items must be ranked with the filter_list parameter,
        in this case ONLY items in the filter_list will be ranked.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, ALL unrated items will be ranked.

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
        raise NotImplementedError
