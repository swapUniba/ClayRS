from __future__ import annotations
import abc
import gc
import itertools
from copy import deepcopy
from itertools import chain
from typing import List, TYPE_CHECKING, Optional, Any, Set, Tuple
import pandas as pd

from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.validation import check_is_fitted
import numpy as np
import torch

from clayrs.recsys.content_based_algorithm.exceptions import UserSkipAlgFit
from clayrs.recsys.methodology import Methodology
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_iterator_parallel

if TYPE_CHECKING:
    from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
        CombiningTechnique
    from clayrs.content_analyzer.ratings_manager.ratings import Ratings
    from clayrs.content_analyzer.content_representation.content import Content
    from clayrs.recsys.algorithm import Algorithm

from clayrs.recsys.content_based_algorithm.contents_loader import LoadedContentsDict


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
    __slots__ = ('item_field', 'threshold', '_transformer')

    def __init__(self, item_field: dict, threshold: float):
        self.item_field: dict = self._bracket_representation(item_field)
        self.threshold: float = threshold
        self._transformer = DictVectorizer(sparse=False, sort=False)

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
    def _calc_mean_user_threshold(user_ratings: List[Interaction]):
        """
        Private method which simply calculates the average rating by the user given its ratings
        """
        return np.nanmean([interaction.score for interaction in user_ratings])

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

    def fuse_representations(self, X: list, embedding_combiner: CombiningTechnique, as_array: bool = False):
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
        if any(not isinstance(rep, (dict, np.ndarray, (int, float), sparse.csc_matrix, torch.Tensor)) for rep in X[0]):
            raise ValueError("You can only use representations of type: {numeric, embedding, tfidf}")

        # We check if there are dicts as representation in the first element of X,
        # since the representations are the same for all elements in X we can check
        # for dicts only in one element
        need_vectorizer = any(isinstance(rep, dict) for rep in X[0])

        if need_vectorizer:
            # IF the transformer is not fitted then we are training the model
            try:
                check_is_fitted(self._transformer)
            except NotFittedError:
                X_dicts = [rep for item in X for rep in item if isinstance(rep, dict)]
                self._transformer.fit(X_dicts)

        # In every case, we transform the input
        def single_item_fused_gen():
            for item_repr_list in X:
                single_arr = []
                for item_repr in item_repr_list:
                    if need_vectorizer and isinstance(item_repr, dict):
                        item_repr = self._transformer.transform(item_repr)
                        single_arr.append(item_repr.flatten())
                    elif isinstance(item_repr, np.ndarray):
                        item_repr = item_repr.flatten()
                        if item_repr.ndim > 1:
                            item_repr = embedding_combiner.combine(item_repr)

                        single_arr.append(item_repr.flatten())
                    elif isinstance(item_repr, torch.Tensor):
                        item_repr = item_repr.numpy().flatten()

                        if item_repr.ndim > 1:
                            item_repr = embedding_combiner.combine(item_repr)

                        single_arr.append(item_repr.flatten())
                    else:
                        # it's a float
                        single_arr.append(item_repr)

                yield single_arr

        # if a representation used is a sparse matrix, then we use scipy library to concatenate
        # otherwise, if we have all dense arrays, we use numpy. To do this check we consider the representations
        # of the first item
        first_arr = next(single_item_fused_gen())
        if any(isinstance(x, sparse.csc_matrix) for x in first_arr):
            X_vectorized = (sparse.hstack(single_arr) for single_arr in single_item_fused_gen())

            X_vectorized = sparse.vstack(X_vectorized, format='csr')

            if as_array is True:
                X_vectorized = X_vectorized.toarray()
        else:
            X_vectorized = [np.hstack(single_arr) for single_arr in single_item_fused_gen()]

            X_vectorized = np.array(X_vectorized)

        return X_vectorized

    @abc.abstractmethod
    def fit(self, train_set: Ratings, items_directory: str, num_cpus: int = 0) -> Any:
        """
        Abstract method that fits the content-based algorithm.

        Every content based algorithm has a different fit process, it may be needed to fit a classifier,
        to build the centroid of the positive items, to build a query for the index, etc.

        It must be called after the process_rated() method since it uses private attributes calculated
        by said method to fit the algorithm.

        The fitted object will also be stored in a private attribute.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, fit_alg: Any, train_set: Ratings, test_set: Ratings, items_directory: str,
             user_idx_list: Set[int], n_recs: Optional[int], methodology: Methodology,
             num_cpus: int) -> List[np.ndarray]:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All unrated items for the user will be ranked (or only items in the filter list, if specified).

        One can specify which items must be ranked with the `filter_list` parameter,
        in this case ONLY items in the `filter_list` parameter will be ranked.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, **ALL** unrated items will be ranked.

        Args:
            user_ratings: List of Interaction objects for a single user
            available_loaded_items: The LoadedContents interface which contains loaded contents
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            filter_list (list): list of the items to rank, if None all unrated items for the user will be ranked

        Returns:
            List of Interactions object in a descending order w.r.t the 'score' attribute, representing the ranking for
            a single user
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, fit_alg: Any, train_set: Ratings, test_set: Ratings, items_directory: str,
                user_idx_list: Set[int], methodology: Methodology,
                num_cpus: int) -> List[np.ndarray]:
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
            available_loaded_items (LoadedContentsDict): loaded contents interface point to the items directory
            filter_list (list): list of the items to predict, if None all unrated items will be score predicted
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the score predicted
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_rank(self, train_set: Ratings, test_set: Ratings, items_directory: str, user_idx_list: Set[int],
                 n_recs: Optional[int], methodology: Methodology, num_cpus: int,
                 save_fit: bool) -> Tuple[Optional[Any], List[np.ndarray]]:
        raise NotImplementedError

    @abc.abstractmethod
    def fit_predict(self, train_set: Ratings, test_set: Ratings, items_directory: str, user_idx_list: Set[int],
                    methodology: Methodology, num_cpus: int,
                    save_fit: bool) -> Tuple[Optional[Any], List[np.ndarray]]:
        raise NotImplementedError

    def _load_available_contents(self, contents_path: str, items_to_load: set = None):
        return LoadedContentsDict(contents_path, items_to_load, only_representations=self.item_field)

    def __deepcopy__(self, memo):
        # Create a new instance
        cls = self.__class__
        result = cls.__new__(cls)

        # Don't copy self reference
        memo[id(self)] = result

        # Don't copy the cache - if it exists
        if hasattr(self, "_cache"):
            memo[id(self._cache)] = self._cache.__new__(dict)

        # Get all __slots__ of the derived class
        slots = chain.from_iterable(getattr(s, '__slots__', []) for s in self.__class__.__mro__)

        # Deep copy all other attributes
        for var in slots:
            setattr(result, var, deepcopy(getattr(self, var), memo))

        # Return updated instance
        return result


class PerUserCBAlgorithm(ContentBasedAlgorithm):
    __slots__ = ()

    @abc.abstractmethod
    def process_rated(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict):
        """
        Abstract method that processes rated items for the user.

        Every content-based algorithm processes rated items differently, it may be needed to extract features
        from the rated items and label them, extract features only from the positive ones, etc.

        The rated items processed must be stored into a private attribute of the algorithm, later used
        by the fit() method.

        Args:
            user_ratings (pd.DataFrame): DataFrame containing ratings of a single user
            available_loaded_items (LoadedContentsDict): loaded contents interface point to the items directory
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_single_user(self):
        raise NotImplementedError

    @abc.abstractmethod
    def predict_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict,
                            filter_list) -> np.ndarray:
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
            available_loaded_items (LoadedContentsDict): loaded contents interface point to the items directory
            filter_list (list): list of the items to predict, if None all unrated items will be score predicted
        Returns:
            pd.DataFrame: DataFrame containing one column with the items name,
                one column with the score predicted
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict,
                         recs_number, filter_list) -> np.ndarray:
        """
        Rank the top-n recommended items for the user. If the recs_number parameter isn't specified,
        All unrated items for the user will be ranked (or only items in the filter list, if specified).

        One can specify which items must be ranked with the `filter_list` parameter,
        in this case ONLY items in the `filter_list` parameter will be ranked.
        One can also pass items already seen by the user with the filter_list parameter.
        Otherwise, **ALL** unrated items will be ranked.

        Args:
            user_ratings: List of Interaction objects for a single user
            available_loaded_items: The LoadedContents interface which contains loaded contents
            recs_number: number of the top ranked items to return, if None all ranked items will be returned
            filter_list (list): list of the items to rank, if None all unrated items for the user will be ranked

        Returns:
            List of Interactions object in a descending order w.r.t the 'score' attribute, representing the ranking for
            a single user
        """
        raise NotImplementedError

    def fit(self, train_set: Ratings, items_directory: str, num_cpus: int = 1) -> Any:
        """
        Method which will fit the algorithm chosen for each user in the train set passed in the constructor

        If the algorithm can't be fit for some users, a warning message is printed
        """
        def compute_single_fit(user_idx):

            nonlocal count_skipped_user

            try:
                self.process_rated(user_idx, train_set, loaded_items_interface)
                self.fit_single_user()
                user_fit_fns = (self.rank_single_user, self.predict_single_user)
            except UserSkipAlgFit as e:
                # warning_message = str(e) + f"\nNo algorithm will be fit for the user {user_id}"
                # logger.warning(warning_message)
                user_fit_fns = None
                count_skipped_user += 1

            return user_idx, user_fit_fns

        count_skipped_user = 0
        items_to_load = train_set.unique_item_id_column
        all_users = train_set.unique_user_idx_column
        loaded_items_interface = self._load_available_contents(items_directory, items_to_load)

        users_fit_dict = {}
        with get_iterator_parallel(num_cpus,
                                   compute_single_fit, all_users,
                                   progress_bar=True, total=len(all_users)) as pbar:

            pbar.set_description("Fitting algorithm")

            for user_idx, fitted_user_alg in pbar:
                if fitted_user_alg is not None:
                    users_fit_dict[user_idx] = fitted_user_alg

        if count_skipped_user > 0:
            logger.warning(f"{count_skipped_user} users will be skipped because the algorithm chosen "
                           f"could not be fit for them")

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        return users_fit_dict

    def rank(self, users_fit_dict: dict, train_set: Ratings, test_set: Ratings, items_directory: str,
             user_idx_list: Set[int], n_recs: Optional[int], methodology: Methodology,
             num_cpus: int) -> List[np.ndarray]:

        def compute_single_rank(user_idx: int):

            nonlocal count_skipped_user

            filter_list = methodology.filter_single(user_idx, train_set, test_set).astype(int)
            # need to convert back int to str to load serialized items
            filter_list = train_set.item_map.convert_seq_int2str(filter_list)

            user_fit_alg = users_fit_dict.get(user_idx)
            if user_fit_alg is not None:
                # we access [0] since [0] is rank_fn, [1] is pred_fn
                user_fit_alg_rank_fn = user_fit_alg[0]
                user_rank = user_fit_alg_rank_fn(user_idx, test_set, loaded_items_interface,
                                                 recs_number=n_recs, filter_list=filter_list)
            else:
                user_rank = np.array([])
                count_skipped_user += 1
                # logger.warning(f"No algorithm fitted for user {user_idx}! It will be skipped")

            return user_idx, user_rank

        count_skipped_user = 0

        loaded_items_interface = self._load_available_contents(items_directory, set())

        uir_rank_list = []

        with get_iterator_parallel(num_cpus,
                                   compute_single_rank, user_idx_list,
                                   progress_bar=True, total=len(user_idx_list)) as pbar:

            pbar.set_description(f"Loading first items from memory...")
            for user_idx, user_rank in pbar:
                pbar.set_description(f"Computing rank for user {user_idx}")
                uir_rank_list.append(user_rank)

        if count_skipped_user > 0:
            logger.warning(f"{count_skipped_user} users will be skipped because the algorithm chosen "
                           f"was not fit for them")

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        return uir_rank_list

    def predict(self, users_fit_dict: dict, train_set: Ratings, test_set: Ratings, items_directory: str,
                user_idx_list: Set[int], methodology: Methodology, num_cpus: int) -> List[np.ndarray]:

        def compute_single_predict(user_idx: int):

            nonlocal count_skipped_user

            filter_list = methodology.filter_single(user_idx, train_set, test_set).astype(int)
            # need to convert back int to str to load serialized items
            filter_list = train_set.item_map.convert_seq_int2str(filter_list)

            user_fitted_alg = users_fit_dict.get(user_idx)
            if user_fitted_alg is not None:
                # we access [1] since [1] is pred_fn, [0] is rank_fn
                user_fitted_alg_pred_fn = user_fitted_alg[1]
                user_pred = user_fitted_alg_pred_fn(user_idx, train_set, loaded_items_interface, filter_list=filter_list)
            else:
                user_pred = np.array([])
                count_skipped_user += 1
                # logger.warning(f"No algorithm fitted for user {user_id}! It will be skipped")

            return user_idx, user_pred

        count_skipped_user = 0

        loaded_items_interface = self._load_available_contents(items_directory, set())

        uir_pred_list = []

        with get_iterator_parallel(num_cpus,
                                   compute_single_predict, user_idx_list,
                                   progress_bar=True, total=len(user_idx_list)) as pbar:

            pbar.set_description(f"Loading first items from memory...")
            for user_idx, user_pred in pbar:
                pbar.set_description(f"Computing score prediction for user {user_idx}")
                uir_pred_list.append(user_pred)

        if count_skipped_user > 0:
            logger.warning(f"{count_skipped_user} users will be skipped because the algorithm chosen "
                           f"was not fit for them")

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        return uir_pred_list

    def fit_rank(self, train_set: Ratings, test_set: Ratings, items_directory: str, user_idx_list: Set[int],
                 n_recs: Optional[int], methodology: Methodology, num_cpus: int,
                 save_fit: bool) -> Tuple[Optional[dict], List[np.ndarray]]:

        def compute_single_fit_rank(user_idx):

            nonlocal count_skipped_user, users_fit_dict

            try:
                self.process_rated(user_idx, train_set, loaded_items_interface)
                self.fit_single_user()

                if save_fit:
                    users_fit_dict[user_idx] = (self.rank_single_user, self.predict_single_user)

            except UserSkipAlgFit as e:
                # warning_message = str(e) + f"\nThe algorithm can't be fitted for the user {user_id}"
                # logger.warning(warning_message)
                count_skipped_user += 1
                return user_idx, np.array([])

            filter_list = methodology.filter_single(user_idx, train_set, test_set).astype(int)
            # need to convert back int to str to load serialized items
            filter_list = train_set.item_map.convert_seq_int2str(filter_list)

            user_rank = self.rank_single_user(user_idx, test_set, loaded_items_interface,
                                              n_recs, filter_list=filter_list)

            return user_idx, user_rank

        count_skipped_user = 0
        users_fit_dict = {} if save_fit else None

        loaded_items_interface = self._load_available_contents(items_directory, set())

        uir_rank_list = []

        with get_iterator_parallel(num_cpus,
                                   compute_single_fit_rank, user_idx_list,
                                   progress_bar=True, total=len(user_idx_list)) as pbar:

            pbar.set_description(f"Loading first items from memory...")
            for user_idx, user_rank in pbar:
                pbar.set_description(f"Computing fit_rank for user {user_idx}")
                uir_rank_list.append(user_rank)

        if count_skipped_user > 0:
            logger.warning(f"{count_skipped_user} users will be skipped because the algorithm chosen "
                           f"could not be fit for them")

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        return users_fit_dict, uir_rank_list

    def fit_predict(self, train_set: Ratings, test_set: Ratings, items_directory: str, user_idx_list: Set[int],
                    methodology: Methodology, num_cpus: int,
                    save_fit: bool) -> Tuple[Optional[dict], List[np.ndarray]]:

        count_skipped_user = 0
        users_fit_dict = {} if save_fit else None

        def compute_single_fit_predict(user_idx):

            nonlocal count_skipped_user, users_fit_dict

            try:
                self.process_rated(user_idx, train_set, loaded_items_interface)
                self.fit_single_user()

                if save_fit:
                    users_fit_dict[user_idx] = (self.rank_single_user, self.predict_single_user)

            except UserSkipAlgFit as e:
                # warning_message = str(e) + f"\nThe algorithm can't be fitted for the user {user_id}"
                # logger.warning(warning_message)
                count_skipped_user += 1
                return user_idx, np.array([])

            filter_list = methodology.filter_single(user_idx, train_set, test_set).astype(int)
            # need to convert back int to str to load serialized items
            filter_list = train_set.item_map.convert_seq_int2str(filter_list)

            user_pred = self.predict_single_user(user_idx, test_set, loaded_items_interface, filter_list=filter_list)

            return user_idx, user_pred

        loaded_items_interface = self._load_available_contents(items_directory, set())

        uir_pred_list = []

        with get_iterator_parallel(num_cpus,
                                   compute_single_fit_predict, user_idx_list,
                                   progress_bar=True, total=len(user_idx_list)) as pbar:

            pbar.set_description(f"Loading first items from memory...")
            for user_idx, user_rank in pbar:
                pbar.set_description(f"Computing fit_rank for user {user_idx}")
                uir_pred_list.append(user_rank)

        if count_skipped_user > 0:
            logger.warning(f"{count_skipped_user} users will be skipped because the algorithm chosen "
                           f"could not be fit for them")

        # we force the garbage collector after freeing loaded items
        del loaded_items_interface
        gc.collect()

        return users_fit_dict, uir_pred_list
