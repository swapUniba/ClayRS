from __future__ import annotations
import gc
import os
import random
from typing import Any, Set, Optional, Type, Dict, Callable, TYPE_CHECKING, Tuple, List

if TYPE_CHECKING:
    from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
        CombiningTechnique
    from clayrs.recsys.methodology import Methodology

from clayrs.content_analyzer.ratings_manager.ratings import Ratings, StrIntMap
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import Centroid
from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg
from clayrs.recsys.network_based_algorithm.amar.amar_network import AmarDataset, AmarNetwork, SingleSourceAmarNetwork
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_iterator_parallel, get_progbar

import numpy as np

import torch
import torch.nn.functional as fun
import torch.utils.data as data

# TO-DO: maybe unify amar and vbpr so that they both inherit from a Network class?
# some implementations and comments are redundant w.r.t. vbpr implementation but they still need to be
# repeated here (e.g. seed_all or rank)
# main problem is the difference in fields, e.g. item_field_list for AMAR and single field for VBPR


__all__ = ["AmarSingleSource", "AmarDoubleSource"]


class Amar(ContentBasedAlgorithm):
    r"""
    Class that implements recommendation for AMAR (Ask Me Any Rating) neural architectures.
    It's a ranking algorithm, so it can't do score prediction.

    Args:
        network_to_use: AmarNetwork class which will be used to instantiate the related network. It will be instantiated
            using `additional_network_parameters` if specified
        item_field_list: list of dict where the key is the name of the field that contains the content to use, value
            is the representation(s) id(s) that will be used for said item. The value of a field can be a string or
            a list, use a list if you want to use multiple representations for a particular field. It is a list
            of dicts because AMAR architectures are capable of processing data coming from multiple
            different sources (so the first item field will refer to the first information source, the second one
            for the second information source, and so on)
        user_field_list: list of dict where the key is the name of the field that contains the content to use, value
            is the representation(s) id(s) that will be used for said user. The value of a field can be a string or
            a list, use a list if you want to use multiple representations for a particular field. It is a list of dicts
            because AMAR architecture are capable of processing data coming from multiple different
            sources (so the first user field will refer to the first information source, the second on for the
            second information source, and so on). The user_field_list may not be specified or some dict inside of it
            may be None, in such cases, the user representation will be computed as the centroid of the items
            liked by the users (where the item representation used refers to the item field that matches the index of
            the user field representation).

            For example:

                item_field_list = [{'item_field_1': 0}, {'item_field_2': 0}]
                user_field_list = [{'user_field_1': 0}, None]

                The representation with id 0 will be used for both 'item_field_1' and 'user_field_1' for the first
                pair of user-item fields, while for the second one representation with id 0 for 'item_field_2' will be
                used while for the user representation the centroid will be calculated

        epochs: number of training epochs
        threshold: float value which is used to distinguish positive from negative items. If None, it will vary for each
            user, and it will be set to the average rating given by it
        learning_rate: learning rate for the torch optimizer
        train_loss: loss function for the training phase. Default is binary cross entropy loss
        optimizer_class: optimizer torch class for the training phase. It will be instantiated using
            `additional_opt_parameters` if specified
        device: device on which the training will be run. If None and a GPU is available, then the GPU is automatically
            selected as device to use. Otherwise, the cpu is used
        embedding_combiner: `CombiningTechnique` used when embeddings representation must be used, but they are in a
            matrix form instead of a single vector (e.g. WordEmbedding representations have one
            vector for each word). By default, the `Centroid` of the rows of the matrix is computed
        seed: random state which will be used for weight initialization and sampling of the negative example
        additional_network_parameters: kwargs for the network.
        additional_opt_parameters: kwargs for the optimizer. If you specify *learning rate* in this parameter, it will
            be overwritten by the local `learning_rate` parameter
        additional_dl_parameters: kwargs for the dataloader. If you specify *batch size* in this parameter, it will
            be overwritten by the local `batch_size` parameter
    """

    def __init__(self, network_to_use: Type[AmarNetwork],
                 item_field_list: List[dict],
                 batch_size: int, epochs: int,
                 user_field_list: List[dict],
                 threshold: Optional[float] = 0,
                 learning_rate: float = 0.001,
                 train_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = fun.binary_cross_entropy,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 device: str = None,
                 embedding_combiner: CombiningTechnique = Centroid(),
                 seed: int = None,
                 additional_network_parameters: Dict[str, Any] = None,
                 additional_opt_parameters: Dict[str, Any] = None,
                 additional_dl_parameters: Dict[str, Any] = None,
                 custom_network_weights: Dict[str, np.array] = None):

        super().__init__({}, threshold)

        if additional_opt_parameters is None:
            additional_opt_parameters = {}

        if additional_dl_parameters is None:
            additional_dl_parameters = {}

        if additional_network_parameters is None:
            additional_network_parameters = {}

        additional_opt_parameters["lr"] = learning_rate
        additional_dl_parameters["batch_size"] = batch_size

        for i, item_field in enumerate(item_field_list):
            item_field_list[i] = self._bracket_representation(item_field)

        for i, user_field in enumerate(user_field_list):
            if user_field is None:
                user_field = {}
            user_field_list[i] = self._bracket_representation(user_field)

        self.item_field_list = item_field_list
        self.user_field_list = user_field_list

        self.device = device if device is not None else "cuda:0" if torch.cuda.is_available() else "cpu"

        self.epochs = epochs
        self.train_loss = train_loss
        self.train_optimizer = optimizer_class
        self.train_optimizer_parameters = additional_opt_parameters

        self._embedding_combiner = embedding_combiner

        self.seed = seed
        self.additional_network_parameters = additional_network_parameters
        self.dl_parameters = additional_dl_parameters
        self.custom_network_weights = custom_network_weights
        self.network = network_to_use

    def _seed_all(self):

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

            cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")

            if cublas_config == ":16:8" or cublas_config == ":4096:8":
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    def _make_ratings_implicit(self, train_set: Ratings) -> Ratings:
        """
        Private function which converts explicit feedback to implicit and returns a `Ratings` object

        An interaction is considered positive IF:
            * interactions with `score >= self.threshold` if `threshold` was set,
              `score >= mean_rating_u` for each user u otherwise
        """

        logger.info("Turning interactions to implicit...")
        # constant threshold for all users

        if self.threshold is not None:

            positive_items_idxs = train_set.score_column >= self.threshold
            negative_items_idxs = train_set.score_column < self.threshold

        # the threshold will vary for each user (its mean rating will be used)
        else:

            user_threshold = []

            for user_idx in train_set.unique_user_idx_column:
                user_interactions = train_set.get_user_interactions(user_idx).copy()
                mean_threshold = np.nanmean(user_interactions[:, 2])
                user_threshold.append(mean_threshold)

            user_threshold = np.array(user_threshold)
            positive_items_idxs = train_set.score_column >= user_threshold[train_set.user_idx_column]
            negative_items_idxs = np.invert(positive_items_idxs)

        uir = train_set.uir.copy()

        uir[:, 2][positive_items_idxs] = 1
        uir[:, 2][negative_items_idxs] = 0

        implicit_train_set = Ratings.from_uir(uir, train_set.user_map, train_set.item_map)

        return implicit_train_set

    def _load_contents_features(self, map: StrIntMap, directory: str, content_field: dict) -> np.ndarray:
        """
        Private function which loads into memory contents (items or users) features
        """

        loaded_contents_interface = self._load_available_contents(directory, set(), None)

        all_content_features = []

        with get_progbar(map) as pbar:

            pbar.set_description("Loading features from serialized contents...")

            for i, content_id in enumerate(pbar):

                content = loaded_contents_interface.get(content_id, content_field, throw_away=True)

                if content is not None:
                    content_features = self.fuse_representations([self.extract_features_item(content, content_field)],
                                                                 self._embedding_combiner, as_array=True)

                    all_content_features.append(content_features)
                else:
                    all_content_features.append(None)

        try:
            first_not_none_element = next(item for item in all_content_features if item is not None)
        except StopIteration:
            raise FileNotFoundError("No contents were loaded!") from None

        for i, content_feature in enumerate(all_content_features):
            if content_feature is None:
                all_content_features[i] = np.zeros(shape=first_not_none_element.shape)

        all_content_features = np.vstack(all_content_features)
        all_content_features = all_content_features.astype(np.float32)

        del loaded_contents_interface
        gc.collect()

        return all_content_features

    def _combine_items_features_for_users(self, implicit_train_set: Ratings, items_features: np.ndarray):
        """
        Private function which computes for each user the centroid for the items that they liked

        The centroids are then returned in a 2D numpy array where each row corresponds to a centroid and is associated
        to a user
        """
        users_features = []

        for user, _ in enumerate(implicit_train_set.user_map):
            interactions = implicit_train_set.get_user_interactions(user)
            positive_interactions = interactions[interactions[:, 2] == 1.]

            if len(positive_interactions) != 0:
                user_features = self._embedding_combiner.combine(items_features[positive_interactions[:, 1].astype(int)])
                users_features.append(user_features)
            else:
                user_features = np.zeros(items_features[0].shape[0])
                users_features.append(user_features)

        return np.vstack(users_features)

    def fit(self, train_set: Ratings, items_directory: str, users_directory: str = None, num_cpus: str = 0) -> AmarNetwork:
        """
        Method which will fit the Amar algorithm via neural training with torch

        Args:
            train_set: `Ratings` object which contains the train set of each user
            items_directory: Path where complexly represented items are serialized by the Content Analyzer
            users_directory: Path where complexly represented users are serialized by the Content Analyzer
            num_cpus: number of processors that must be reserved for the method. If set to `0`, all cpus available will
                be used. Be careful though: multiprocessing in python has a substantial memory overhead!

        Returns:
            A fit AmarNetwork object (torch module which implements an Amar neural network)
        """

        implicit_train_set = self._make_ratings_implicit(train_set)

        items_features = []
        users_features = []

        for item_field in self.item_field_list:
            items_features.append(torch.from_numpy(self._load_contents_features(implicit_train_set.item_map, items_directory, item_field)))

        # load user features if specified, otherwise compute for each user the centroid of its positive items
        for i, user_field in enumerate(self.user_field_list):
            if len(user_field) != 0:
                users_features.append(torch.from_numpy(self._load_contents_features(implicit_train_set.user_map, users_directory, user_field)))
            else:
                users_features.append(torch.from_numpy(self._combine_items_features_for_users(implicit_train_set, items_features[i])))

        self._seed_all()

        model = self.network(items_features=items_features,
                             users_features=users_features,
                             device=self.device,
                             custom_weights=self.custom_network_weights,
                             **self.additional_network_parameters).float()

        optimizer = self.train_optimizer(model.parameters(), **self.train_optimizer_parameters)

        train_dataset = AmarDataset(implicit_train_set)

        train_dl = torch.utils.data.DataLoader(train_dataset, **self.dl_parameters)

        model.train()

        logger.info("Starting training!")

        for epoch in range(self.epochs):

            train_loss = 0

            with get_progbar(train_dl) as pbar:

                pbar.set_description(f"Starting {epoch + 1}/{self.epochs} epoch...")

                epoch_losses = []
                weights_sum = 0

                for i, batch in enumerate(pbar):
                    optimizer.zero_grad()

                    user_idx = batch[0].int()
                    item_idx = batch[1].int()

                    model_input = (
                        user_idx,
                        item_idx
                    )

                    score = model(model_input)
                    loss = self.train_loss(score.flatten().to(self.device), batch[2].to(self.device).float())

                    train_loss += loss.item()
                    epoch_losses.append(loss.item() * len(batch[0]))
                    weights_sum += len(batch[0])

                    loss.backward()
                    optimizer.step()

                    pbar.set_description(f'[Epoch {epoch + 1}/{self.epochs}, '
                                         f'Batch {i + 1}/{len(train_dl)}, '
                                         f'Loss: {sum(epoch_losses) / weights_sum:.4f}]')

        logger.info("Training complete!")
        logger.info("Done!")

        model.eval()

        return model

    def rank(self, fit_alg: AmarNetwork, train_set: Ratings, test_set: Ratings, items_directory: str,
             user_idx_list: Set[int], n_recs: Optional[int], methodology: Methodology,
             num_cpus: int) -> List[np.ndarray]:
        """
        Method used to calculate ranking for all users in `user_idx_list` parameter.
        You must first call the `fit()` method ***before*** you can compute the ranking.
        The `user_idx_list` parameter should contain users with mapped to their integer!

        The representation of the fit Amar algorithm is a `AmarNetwork` object (torch module which implements a
        Amar neural network)

        If the `n_recs` is specified, then the rank will contain the top-n items for the users.
        Otherwise, the rank will contain all unrated items of the particular users.

        Via the `methodology` parameter you can perform different candidate item selection. By default, the
        `TestRatingsMethodology()` is used: so, for each user, items in its test set only will be ranked

        Args:
            fit_alg: a fit `AmarNetwork` object (torch module which implements the Amar neural network)
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

        def compute_single_rank(user_idx):
            filter_list = methodology.filter_single(user_idx, train_set, test_set)
            user_rank = fit_alg.return_scores(np.full(len(filter_list), user_idx), filter_list)
            user_uir = np.array((
                np.full(len(user_rank), user_idx),
                filter_list,
                user_rank.numpy()
            )).T
            # items are not sorted so we sort them (to have descending order, we invert the values of the user uir
            # score column
            sorted_user_uir = user_uir[(-user_uir[:, 2]).argsort()]
            sorted_user_uir = sorted_user_uir[:n_recs]

            return user_idx, sorted_user_uir

        fit_alg.eval()

        methodology.setup(train_set, test_set)

        uir_rank_list = []
        with get_iterator_parallel(num_cpus,
                                   compute_single_rank, user_idx_list,
                                   progress_bar=True, total=len(user_idx_list)) as pbar:
            for user_idx, user_rank in pbar:
                pbar.set_description(f"Computing rank for user {user_idx}")
                uir_rank_list.append(user_rank)

        return uir_rank_list

    def predict(self, fit_alg: AmarNetwork, train_set: Ratings, test_set: Ratings, items_directory: str,
                user_idx_list: Set[int], methodology: Methodology,
                num_cpus: int) -> List[np.ndarray]:

        raise NotPredictionAlg("Amar is not a Score Prediction Algorithm!")

    def fit_rank(self, train_set: Ratings, test_set: Ratings, items_directory: str, user_idx_list: Set[int],
                 n_recs: Optional[int], methodology: Methodology, num_cpus: int, save_fit: bool,
                 users_directory: str = None) -> Tuple[Optional[AmarNetwork], List[np.ndarray]]:
        """
        Method used to both fit and calculate ranking for all users in `user_idx_list` parameter.
        The algorithm will first be fit considering all users in the `user_idx_list` which should contain user id
        mapped to their integer!

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
            users_directory: Path where complexly represented users are serialized by the Content Analyzer

        Returns:
            The first value is the fit Amar algorithm (could be None if `save_fit == False`)

            The second value is a list of predicted uir matrices all sorted in a decreasing order w.r.t.
                the ranking scores
        """
        amar_fit = self.fit(train_set, items_directory, users_directory=users_directory, num_cpus=num_cpus)
        rank = self.rank(amar_fit, train_set, test_set, items_directory, user_idx_list, n_recs, methodology, num_cpus)

        amar_fit = amar_fit if save_fit else None

        return amar_fit, rank

    def fit_predict(self, train_set: Ratings, test_set: Ratings, items_directory: str, user_idx_list: Set[int],
                    methodology: Methodology, num_cpus: int, save_fit: bool, users_directory: str = None) -> Tuple[Optional[AmarNetwork], List[np.ndarray]]:

        raise NotPredictionAlg("Amar is not a Score Prediction Algorithm!")

    def __repr__(self):
        return f"{self}(network={self.network}, item_fields={self.item_field_list}, user_fields={self.user_field_list}, " \
               f"batch_size={self.dl_parameters['batch_size']}, epochs={self.epochs}, threshold={self.threshold}, " \
               f"additional_opt_parameters={self.train_optimizer_parameters['lr']}, " \
               f"train_loss={repr(self.train_loss)}, optimizer_class={repr(self.train_optimizer)}, " \
               f"device={self.device}, embedding_combiner={repr(self._embedding_combiner)}, " \
               f"seed={self.seed}, " \
               f"additional_opt_parameters={ {key: val for key, val in self.dl_parameters.items() if key != 'lr'} }, " \
               f"additional_dl_parameters={ {key: val for key, val in self.dl_parameters.items() if key != 'batch_size'} })"


class AmarSingleSource(Amar):
    r"""
    Class that implements recommendation for AMAR (Ask Me Any Rating) neural architectures which only use
    a single source of information.
    It's a ranking algorithm, so it can't do score prediction.

    Args:
        network_to_use: AmarNetwork class which will be used to instantiate the related network. It will be instantiated
            using `additional_network_parameters` if specified
        item_field: dict where the key is the name of the field that contains the content to use, value
            is the representation(s) id(s) that will be used for said item. The value of a field can be a string or
            a list, use a list if you want to use multiple representations for a particular field
        batch_size: dimension of each batch of the torch dataloader for the features
        epochs: number of training epochs
        user_field: dict where the key is the name of the field that contains the content to use, value
            is the representation(s) id(s) that will be used for said user. The value of a field can be a string or
            a list, use a list if you want to use multiple representations for a particular field. If None, the centroid
            representation of the items liked by the user will be computed for each user
        threshold: float value which is used to distinguish positive from negative items. If None, it will vary for each
            user, and it will be set to the average rating given by it
        learning_rate: learning rate for the torch optimizer
        train_loss: loss function for the training phase. Default is binary cross entropy loss
        optimizer_class: optimizer torch class for the training phase. It will be instantiated using
            `additional_opt_parameters` if specified
        device: device on which the training will be run. If None and a GPU is available, then the GPU is automatically
            selected as device to use. Otherwise, the cpu is used
        embedding_combiner: `CombiningTechnique` used when embeddings representation must be used, but they are in a
            matrix form instead of a single vector (e.g. WordEmbedding representations have one
            vector for each word). By default, the `Centroid` of the rows of the matrix is computed
        seed: random state which will be used for weight initialization and sampling of the negative example
        additional_network_parameters: kwargs for the network.
        additional_opt_parameters: kwargs for the optimizer. If *learning rate* is specified in this parameter, it will
            be overwritten by the local `learning_rate` parameter
        additional_dl_parameters: kwargs for the dataloader. If *batch size* is specified in this parameter, it will
            be overwritten by the local `batch_size` parameter

    """

    def __init__(self, network_to_use: Type[SingleSourceAmarNetwork],
                 item_field: dict,
                 batch_size: int, epochs: int,
                 user_field: dict = None,
                 threshold: Optional[float] = 0,
                 learning_rate: float = 0.001,
                 train_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = fun.binary_cross_entropy,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 device: str = None,
                 embedding_combiner: CombiningTechnique = Centroid(),
                 seed: int = None,
                 additional_network_parameters: Dict[str, Any] = None,
                 additional_opt_parameters: Dict[str, Any] = None,
                 additional_dl_parameters: Dict[str, Any] = None,
                 custom_network_weights: Dict[str, np.array] = None):

        super().__init__(network_to_use, [item_field], batch_size, epochs, [user_field], threshold, learning_rate,
                         train_loss, optimizer_class, device, embedding_combiner, seed,
                         additional_network_parameters, additional_opt_parameters, additional_dl_parameters,
                         custom_network_weights)

    def __str__(self):
        return "AmarSingleSource"


class AmarDoubleSource(Amar):
    r"""
    Class that implements recommendation for AMAR (Ask Me Any Rating) neural architectures which use two
    sources of information.
    It's a ranking algorithm, so it can't do score prediction.

    Args:
        network_to_use: AmarNetwork class which will be used to instantiate the related network. It will be instantiated
            using `additional_network_parameters` if specified
        first_item_field: dict for the first item information source where the key is the name of the field that
            contains the content to use, value is the representation(s) id(s) that will be used for said item.
            The value of a field can be a string or a list, use a list if you want to use multiple representations for
            a particular field
        second_item_field: dict for the second item information source where the key is the name of the field that
            contains the content to use, value is the representation(s) id(s) that will be used for said item.
            The value of a field can be a string or a list, use a list if you want to use multiple representations for
            a particular field
        batch_size: dimension of each batch of the torch dataloader for the features
        epochs: number of training epochs
        first_user_field: dict for the first user information source where the key is the name of the field that
            contains the content to use, value is the representation(s) id(s) that will be used for said user.
            The value of a field can be a string or a list, use a list if you want to use multiple representations for
            a particular field. If None, the centroid representation of the items liked by the user will be computed for
            each user
        second_user_field: dict for the second user information source where the key is the name of the field that
            contains the content to use, value is the representation(s) id(s) that will be used for said user.
            The value of a field can be a string or a list, use a list if you want to use multiple representations for
            a particular field. If None, the centroid representation of the items liked by the user will be computed for
            each user
        threshold: float value which is used to distinguish positive from negative items. If None, it will vary for each
            user, and it will be set to the average rating given by it
        learning_rate: learning rate for the torch optimizer
        train_loss: loss function for the training phase. Default is binary cross entropy loss
        optimizer_class: optimizer torch class for the training phase. It will be instantiated using
            `additional_opt_parameters` if specified
        device: device on which the training will be run. If None and a GPU is available, then the GPU is automatically
            selected as device to use. Otherwise, the cpu is used
        embedding_combiner: `CombiningTechnique` used when embeddings representation must be used, but they are in a
            matrix form instead of a single vector (e.g. WordEmbedding representations have one
            vector for each word). By default, the `Centroid` of the rows of the matrix is computed
        seed: random state which will be used for weight initialization and sampling of the negative example
        additional_network_parameters: kwargs for the network.
        additional_opt_parameters: kwargs for the optimizer. If *learning rate* is specified in this parameter, it will
            be overwritten by the local `learning_rate` parameter
        additional_dl_parameters: kwargs for the dataloader. If *batch size* is specified in this parameter, it will
            be overwritten by the local `batch_size` parameter
    """

    def __init__(self, network_to_use: Type[AmarNetwork],
                 first_item_field: dict,
                 second_item_field: dict,
                 batch_size: int, epochs: int,
                 first_user_field: dict = None,
                 second_user_field: dict = None,
                 threshold: Optional[float] = 0,
                 learning_rate: float = 0.001,
                 train_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = fun.binary_cross_entropy,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 device: str = None,
                 embedding_combiner: CombiningTechnique = Centroid(),
                 seed: int = None,
                 additional_network_parameters: Dict[str, Any] = None,
                 additional_opt_parameters: Dict[str, Any] = None,
                 additional_dl_parameters: Dict[str, Any] = None,
                 custom_network_weights: Dict[str, np.array] = None):

        super().__init__(network_to_use, [first_item_field, second_item_field], batch_size, epochs,
                         [first_user_field, second_user_field], threshold, learning_rate,
                         train_loss, optimizer_class, device, embedding_combiner, seed,
                         additional_network_parameters, additional_opt_parameters, additional_dl_parameters,
                         custom_network_weights)

    def __str__(self):
        return "AmarDoubleSource"

