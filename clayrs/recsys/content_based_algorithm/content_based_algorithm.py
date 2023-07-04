from __future__ import annotations
import abc
import gc
from copy import deepcopy
from itertools import chain
from typing import List, TYPE_CHECKING, Optional, Any, Set, Tuple, Dict, Callable

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
        used for the said item. The value of a field can be a string/int or a list,
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

        Examples:

            >>> item_field = {'Plot': 0, 'Genre': ['tfidf', 1]}
            >>> print(ContentBasedAlgorithm._bracket_representation(item_field))
            {'Plot': [0], 'Genre': ['tfidf', 1]}

        Args:
            item_field: dict that contain values that may need to be bracketed

        Returns:
            The item_field passed with all values inside a list
        """
        for field in item_field:
            if not isinstance(item_field[field], list):
                item_field[field] = [item_field[field]]

        return item_field

    def extract_features_item(self, item: Content):
        """
        Function that extracts the feature of a loaded item using the item_field parameter passed in the
        constructor.

        It extracts only the chosen representations of the chosen fields in the item loaded

        * with `item_field = {'Plot': [0], 'Genre': ['tfidf', 1]}`, the function will extract
        only the representation with `0` as internal id for the field `Plot` and two representations
        for the field `Genre`: one with `tfidf` as *external id* and the other with `1` as *internal id*

        Args:
            item: item loaded of which we need to extract its feature

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
            X: list that contains representations of the items
            embedding_combiner: combining technique in case there are multiple
                vectors with different row size
            as_array: if True, result will always be a np.array regardless of the input (e.g. if input are scipy sparse
                matrices, output is a scipy sparse matrix)
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
                        item_repr = self._transformer.transform(item_repr).squeeze()

                    elif isinstance(item_repr, np.ndarray):
                        item_repr = item_repr.squeeze()
                        if item_repr.ndim > 1:
                            item_repr = embedding_combiner.combine(item_repr).squeeze()

                    elif isinstance(item_repr, torch.Tensor):
                        item_repr = item_repr.numpy().squeeze()

                        if item_repr.ndim > 1:
                            item_repr = embedding_combiner.combine(item_repr).squeeze()

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

        Every content based algorithm has a different fit process, it may be needed to fit a classifier for each user,
        to build the centroid of the positive items of each user, to train a neural network for the entire system, etc.

        Args:
            train_set: Ratings object containing interactions between users and items
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            num_cpus: number of processors that must be reserved for the method

        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank(self, fit_alg: Any, train_set: Ratings, test_set: Ratings, items_directory: str,
             user_idx_list: Set[int], n_recs: Optional[int], methodology: Methodology,
             num_cpus: int) -> List[np.ndarray]:
        """
        Abstract method for ranking the top-n recommended items for the user.
        If the recs_number parameter isn't specified, all ranked items will be returned

        Any CB algorithm needs to know where the items are serialized by the content analyzer, hence the
        `items_directory` parameter

        Remember that `user_idx_list` should contain user id mapped to their integer! Not their string representation!

        Args:
            fit_alg: Result object returned by the `fit()` function of `self` (the chosen algorithm)
            train_set: Ratings object containing interactions between users and items
            test_set: Ratings object which represents the ground truth of the split considered
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            user_idx_list: Set of user idx (int representation) for which a recommendation list must be generated.
                Users should be represented with their mapped integer!
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user.
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method

        Returns:
            List of uir matrices for each user, where each uir contains predicted interactions between users and unseen
                items sorted in a descending way w.r.t. the third dimension which is the ranked score
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, fit_alg: Any, train_set: Ratings, test_set: Ratings, items_directory: str,
                user_idx_list: Set[int], methodology: Methodology,
                num_cpus: int) -> List[np.ndarray]:
        """
        Abstract method that predicts the rating which a user would give to items
        If the algorithm is not a PredictionScore Algorithm, implement this method like this:

        ```python
        def predict():
            raise NotPredictionAlg
        ```

        Any CB algorithm needs to know where the items are serialized by the content analyzer, hence the
        `items_directory` parameter

        Remember that `user_idx_list` should contain user id mapped to their integer! Not their string representation!

        Args:
            fit_alg: Result object returned by the `fit()` function of `self` (the chosen algorithm)
            train_set: Ratings object containing interactions between users and items
            test_set: Ratings object which represents the ground truth of the split considered
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            user_idx_list: Set of user idx (int representation) for which a recommendation list must be generated.
                Users should be represented with their mapped integer!
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method

        Returns:
            List of uir matrices for each user, where each uir contains predicted interactions between users and unseen
                items
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_rank(self, train_set: Ratings, test_set: Ratings, items_directory: str, user_idx_list: Set[int],
                 n_recs: Optional[int], methodology: Methodology, num_cpus: int,
                 save_fit: bool) -> Tuple[Optional[Any], List[np.ndarray]]:
        """
        Method used to both fit and calculate ranking for all users in `user_idx_list` parameter.
        The algorithm will first be fit for each user in the `user_idx_list` which should contain user id
        mapped to their integer!

        Any CB algorithm needs to know where the items are serialized by the content analyzer, hence the
        `items_directory` parameter

        With the `save_fit` parameter you can specify if you need the function to return the algorithm fit (in case
        you want to perform multiple calls to the `predict()` or `rank()` function). If set to True, the first value
        returned by this function will be the fit algorithm and the second will be the list of uir matrices with
        predictions for each user.
        Otherwise, if `save_fit` is False, the first value returned by this function will be `None`

        Args:
            train_set: Ratings object containing interactions between users and items
            test_set: Ratings object which represents the ground truth of the split considered
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            user_idx_list: Set of user idx (int representation) for which a recommendation list must be generated.
                Users should be represented with their mapped integer!
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user.
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method
            save_fit: Boolean value which let you choose if the fit algorithm should be saved and returned by this
                function. If True, the first value returned by this function is the fit algorithm. Otherwise, the first
                value will be None. The second value is always the list of predicted uir matrices

        Returns:
            The first value is the fit VBPR algorithm (could be None if `save_fit == False`)

            The second value is a list of predicted uir matrices all sorted in a decreasing order w.r.t.
                the ranking scores
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_predict(self, train_set: Ratings, test_set: Ratings, items_directory: str, user_idx_list: Set[int],
                    methodology: Methodology, num_cpus: int,
                    save_fit: bool) -> Tuple[Optional[Any], List[np.ndarray]]:
        """
        Method used to both fit and calculate score prediction for all users in `user_idx_list` parameter.
        The algorithm will first be fit for each user in the `user_idx_list` which should contain user id
        mapped to their integer!

        If the algorithm is not a PredictionScore Algorithm, implement this method like this:

        ```python
        def fit_predict():
            raise NotPredictionAlg()
        ```

        Any CB algorithm needs to know where the items are serialized by the content analyzer, hence the
        `items_directory` parameter

        With the `save_fit` parameter you can specify if you need the function to return the algorithm fit (in case
        you want to perform multiple calls to the `predict()` or `rank()` function). If set to True, the first value
        returned by this function will be the fit algorithm and the second will be the list of uir matrices with
        predictions for each user.
        Otherwise, if `save_fit` is False, the first value returned by this function will be `None`

        Args:
            train_set: Ratings object containing interactions between users and items
            test_set: Ratings object which represents the ground truth of the split considered
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            user_idx_list: Set of user idx (int representation) for which a recommendation list must be generated.
                Users should be represented with their mapped integer!
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`
            num_cpus: number of processors that must be reserved for the method
            save_fit: Boolean value which let you choose if the fit algorithm should be saved and returned by this
                function. If True, the first value returned by this function is the fit algorithm. Otherwise, the first
                value will be None. The second value is always the list of predicted uir matrices

        Returns:
            The first value is the fit algorithm (could be None if `save_fit == False`)

            The second value is a list of predicted uir matrices
        """
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
    """
    Abstract class for any CB algorithm that is fit for each user, rather than being fit for the whole system once

    Think of the `CentroidVector` algorithm: it builds a centroid for each user!

    This class has several concrete methods so that you can easily extend and add several per-user cb algorithm without
    implementing many abstract methods!
    """
    __slots__ = ()

    @abc.abstractmethod
    def process_rated(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict):
        """
        Abstract method that processes rated items for the single user.

        Every content-based algorithm processes rated items differently, it may be needed to extract features
        from the rated items and label them, extract features only from the positive ones, etc.

        The rated items processed must be stored into a private attribute of the algorithm, later used
        by the fit() method.

        Args:
            user_idx: Mapped integer of the active user (the user for which we must fit the algorithm)
            train_ratings: `Ratings` object which contains the train set of each user
            available_loaded_items: The LoadedContents interface which contains loaded contents
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_single_user(self):
        """
        Abstract method that fits the content-based algorithm for a single user.

        Every content based algorithm has a different fit process, it may be needed to fit a classifier,
        to build the centroid of the positive items of the user, etc.

        It must be called after the process_rated() method since it uses private attributes calculated
        by said method to fit the algorithm.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict,
                            filter_list: List[str]) -> np.ndarray:
        """
        Predicts how much a user will like unrated items.

        The filter list parameter is usually the result of the `filter_single()` method of a `Methodology` object, and
        is a list of items represented with their string ids. Must be necessarily strings and not their mapped integer
        since items are serialized following their string representation!

        Args:
            user_idx: Mapped integer of the active user
            train_ratings: `Ratings` object which contains the train set of each user
            available_loaded_items: The LoadedContents interface which contains loaded contents
            filter_list: list of the items to rank. Should contain string item ids

        Returns:
            uir matrix for a single user containing user and item idxs (integer representation) with the predicted score
                as third dimension
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rank_single_user(self, user_idx: int, train_ratings: Ratings, available_loaded_items: LoadedContentsDict,
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
        raise NotImplementedError

    def fit(self, train_set: Ratings, items_directory: str, num_cpus: int = 1) -> Dict[int, Tuple[Callable, Callable]]:
        """
        Method which will fit the algorithm chosen for each user in the `train_set` parameter

        If the algorithm can't be fit for some users, a warning message is printed showing the number of users
        for which the alg couldn't be fit

        Args:
            train_set: `Ratings` object which contains the train set of each user
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Returns:
            A dictionary with users idxs (int representation) are keys and tuples containing (`rank_fn`,
                `predict_fn`) are values. In this dictionary only users for which the *fit* process could be performed
                appear!
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
        """
        Method used to calculate ranking for all users in `user_idx_list` parameter.
        You must first call the `fit()` method ***before*** you can compute the ranking.
        The `user_idx_list` parameter should contain users with mapped to their integer!

        The representation of the fit per-user cb algorithm is a dictionary where users idxs are keys and
        a tuples containing `rank_fn` and `predict_fn` are values

        If the `n_recs` is specified, then the rank will contain the top-n items for the users.
        Otherwise, the rank will contain all unrated items of the particular users.

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be ranked

        If the algorithm was not fit for some users, they will be skipped and a warning message is printed showing the
        number of users for which the alg couldn't produce a ranking

        Args:
            users_fit_dict: dictionary with users idxs (int representation) are keys and tuples containing (`rank_fn`,
                `predict_fn`) are values. In this dictionary only users for which the *fit* process could be performed
                appear!
            train_set: `Ratings` object which contains the train set of each user
            test_set: Ratings object which represents the ground truth of the split considered
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            user_idx_list: Set of user idx (int representation) for which a recommendation list must be generated.
                Users should be represented with their mapped integer!
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user. Default is 10 (top-10 for each user
                will be computed)
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`. If None, AllItemsMethodology() will be used
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Returns:
            List of uir matrices for each user, where each uir contains predicted interactions between users and unseen
                items sorted in a descending way w.r.t. the third dimension which is the ranked score
        """

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
        """
        Method used to calculate score prediction for all users in `user_idx_list` parameter.
        You must first call the `fit()` method ***before*** you can compute the ranking.
        The `user_idx_list` parameter should contain users with mapped to their integer!

        The representation of the fit per-user cb algorithm is a dictionary where users idxs are keys and
        a tuples containing `rank_fn` and `predict_fn` are values

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be ranked

        If the algorithm was not fit for some users, they will be skipped and a warning message is printed showing the
        number of users for which the alg couldn't produce a ranking

        Args:
            users_fit_dict: dictionary with users idxs (int representation) are keys and tuples containing (`rank_fn`,
                `predict_fn`) are values. In this dictionary only users for which the *fit* process could be performed
                appear!
            train_set: `Ratings` object which contains the train set of each user
            test_set: Ratings object which represents the ground truth of the split considered
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            user_idx_list: Set of user idx (int representation) for which a recommendation list must be generated.
                Users should be represented with their mapped integer!
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`. If None, AllItemsMethodology() will be used
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Returns:
            List of uir matrices for each user, where each uir contains predicted interactions between users and unseen
                items
        """

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
        """
        Method used to both fit and calculate ranking for all users in `user_idx_list` parameter.
        The algorithm will first be fit for each user in the `user_idx_list` which should contain user id
        mapped to their integer!

        Any CB algorithm needs to know where the items are serialized by the content analyzer, hence the
        `items_directory` parameter

        With the `save_fit` parameter you can specify if you need the function to return the algorithm fit (in case
        you want to perform multiple calls to the `predict()` or `rank()` function). If set to True, the first value
        returned by this function will be the fit algorithm and the second will be the list of uir matrices with
        predictions for each user.
        Otherwise, if `save_fit` is False, the first value returned by this function will be `None`

        Args:
            train_set: `Ratings` object which contains the train set of each user
            test_set: Ratings object which represents the ground truth of the split considered
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            user_idx_list: Set of user idx (int representation) for which a recommendation list must be generated.
                Users should be represented with their mapped integer!
            n_recs: Number of the top items that will be present in the ranking of each user.
                If `None` all candidate items will be returned for the user. Default is 10 (top-10 for each user
                will be computed)
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`. If None, AllItemsMethodology() will be used
            save_fit: Boolean value which let you choose if the fit algorithm should be saved and returned by this
                function. If True, the first value returned by this function is the fit algorithm. Otherwise, the first
                value will be None. The second value is always the list of predicted uir matrices
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Returns:
            A tuple where the first value is the fit VBPR algorithm (could be None if `save_fit == False`), the second
            one is a list of predicted uir matrices all sorted in a decreasing order w.r.t. the ranking scores
        """

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
        """
        Method used to both fit and calculate score prediction for all users in `user_idx_list` parameter.
        The algorithm will first be fit for each user in the `user_idx_list` which should contain user id
        mapped to their integer!

        If the algorithm is not a PredictionScore Algorithm, implement this method like this:

        ```python
        def fit_predict():
            raise NotPredictionAlg()
        ```

        Any CB algorithm needs to know where the items are serialized by the content analyzer, hence the
        `items_directory` parameter

        With the `save_fit` parameter you can specify if you need the function to return the algorithm fit (in case
        you want to perform multiple calls to the `predict()` or `rank()` function). If set to True, the first value
        returned by this function will be the fit algorithm and the second will be the list of uir matrices with
        predictions for each user.
        Otherwise, if `save_fit` is False, the first value returned by this function will be `None`

        Args:
            train_set: `Ratings` object which contains the train set of each user
            test_set: Ratings object which represents the ground truth of the split considered
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            user_idx_list: Set of user idx (int representation) for which a recommendation list must be generated.
                Users should be represented with their mapped integer!
            methodology: `Methodology` object which governs the candidate item selection. Default is
                `TestRatingsMethodology`. If None, AllItemsMethodology() will be used
            save_fit: Boolean value which let you choose if the fit algorithm should be saved and returned by this
                function. If True, the first value returned by this function is the fit algorithm. Otherwise, the first
                value will be None. The second value is always the list of predicted uir matrices
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Returns:
            A tuple where the first value is the fit algorithm (could be None if `save_fit == False`), the second
            one is a list of predicted uir matrices
        """

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
