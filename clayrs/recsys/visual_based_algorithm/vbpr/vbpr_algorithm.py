from __future__ import annotations
import gc
import os
import random
from typing import Any, Set, Optional, Type, Dict, Callable, TYPE_CHECKING, Tuple, List

from clayrs.recsys.content_based_algorithm.exceptions import NotPredictionAlg

if TYPE_CHECKING:
    from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
        CombiningTechnique
    from clayrs.recsys.methodology import Methodology

from clayrs.content_analyzer.ratings_manager.ratings import Ratings
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import Centroid
from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm

import numpy as np

import torch
import torch.nn.functional as fun
import torch.utils.data as data

from clayrs.recsys.visual_based_algorithm.vbpr.vbpr_network import VBPRNetwork, TriplesDataset
from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_iterator_parallel, get_progbar

__all__ = ["VBPR"]


class VBPR(ContentBasedAlgorithm):

    def __init__(self, item_field: dict,
                 gamma_dim: int, theta_dim: int, batch_size: int, epochs: int,
                 threshold: Optional[float] = 0,
                 learning_rate: float = 0.005,
                 lambda_w: float = 0.01, lambda_b_pos: float = 0.01, lambda_b_neg: float = 0.001, lambda_e: float = 0,
                 train_loss: Callable[[torch.Tensor], torch.Tensor] = fun.logsigmoid,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 device: str = None,
                 embedding_combiner: CombiningTechnique = Centroid(),
                 normalize: bool = True,
                 seed: int = None,
                 additional_opt_parameters: Dict[str, Any] = None,
                 additional_dl_parameters: Dict[str, Any] = None):

        super().__init__(item_field, threshold)

        if additional_opt_parameters is None:
            additional_opt_parameters = {}

        if additional_dl_parameters is None:
            additional_dl_parameters = {}

        additional_opt_parameters["lr"] = learning_rate
        additional_dl_parameters["batch_size"] = batch_size

        self.device = device if device is not None else "cuda:0" if torch.cuda.is_available() else "cpu"

        self.gamma_dim = gamma_dim
        self.theta_dim = theta_dim

        self.epochs = epochs
        self.train_loss = train_loss
        self.train_optimizer = optimizer_class
        self.train_optimizer_parameters = additional_opt_parameters
        self.normalize = normalize
        self.lambda_w = lambda_w
        self.lambda_b_pos = lambda_b_pos
        self.lambda_b_neg = lambda_b_neg
        self.lambda_e = lambda_e

        self._embedding_combiner = embedding_combiner

        self.seed = seed
        self.dl_parameters = additional_dl_parameters

    def _seed_all(self):

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # most probably these 2 need to be set BEFORE
            # in the environment manually
            os.environ["PYTHONHASHSEED"] = str(self.seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    def _build_only_positive_ratings(self, train_set: Ratings) -> Ratings:

        logger.info("Filtering only positive interactions...")
        # constant threshold for all users
        if self.threshold is not None:

            positive_items_idxs = train_set.score_column >= self.threshold

            if np.count_nonzero(positive_items_idxs) != len(train_set):

                positive_train_set = train_set.uir[positive_items_idxs]
                positive_train_set = Ratings.from_uir(positive_train_set, train_set.user_map, train_set.item_map)
            else:
                positive_train_set = train_set
                logger.info(f"All interactions have score >= than threshold={self.threshold}, "
                            f"no filtering is performed")

        # the threshold will vary for each user (its mean rating will be used)
        else:

            positive_interactions = []

            for user_idx in train_set.unique_user_idx_column:

                user_interactions = train_set.get_user_interactions(user_idx)
                mean_threshold = np.nanmean(user_interactions[:, 2])
                positive_items_idxs = np.where(user_interactions[:, 2] >= mean_threshold)
                positive_interactions.append(user_interactions[positive_items_idxs])

            positive_interactions = np.vstack(positive_interactions)

            positive_train_set = Ratings.from_uir(positive_interactions,
                                                  train_set.user_map,
                                                  train_set.item_map)

        # in both cases:
        #   check if no ratings remains in train set exception is raised
        #   check if some users are missing because no positive items remains for them, warning is issued

        if len(positive_train_set) == 0:
            raise ValueError("Filtering for positive interactions didn't leave any rating at all!")

        diff_len_train = len(train_set.unique_user_idx_column) - len(positive_train_set.unique_user_idx_column)

        if diff_len_train != 0:
            logger.warning(f"{diff_len_train} users were skipped because no positive interaction remained "
                           f"after filtering")

        return positive_train_set

    def _load_items_features(self, train_set: Ratings, items_directory: str) -> np.ndarray:

        loaded_items_interface = self._load_available_contents(items_directory, set())

        items_features = []

        with get_progbar(train_set.item_map) as pbar:
            pbar.set_description("Loading features from serialized items...")

            for i, item_id in enumerate(pbar):

                item = loaded_items_interface.get(item_id, self.item_field, throw_away=True)

                if item is not None:
                    item_features = self.fuse_representations([self.extract_features_item(item)],
                                                              self._embedding_combiner, as_array=True)

                    items_features.append(item_features)
                else:
                    items_features.append(None)

        try:
            first_not_none_element = next(item for item in items_features if item is not None)
        except StopIteration:
            raise FileNotFoundError("No items were loaded!") from None

        for i, item_feature in enumerate(items_features):
            if item_feature is None:
                items_features[i] = np.zeros(shape=first_not_none_element.shape)
        
        items_features = np.vstack(items_features)
        
        if self.normalize is True:
            items_features = items_features - np.min(items_features)
            items_features = items_features / (np.max(items_features) + 1e-10)

        items_features = items_features.astype(np.float32)

        del loaded_items_interface
        gc.collect()

        return items_features

    def fit(self, train_set: Ratings, items_directory: str, num_cpus: int = -1) -> VBPRNetwork:

        def _l2_loss(*tensors):
            l2_loss = 0
            for tensor in tensors:
                l2_loss += tensor.pow(2).sum()
            return l2_loss / 2

        train_set = self._build_only_positive_ratings(train_set)

        items_features = self._load_items_features(train_set, items_directory)

        self._seed_all()

        items_features = torch.tensor(items_features, device=self.device, dtype=torch.float)

        model = VBPRNetwork(n_users=len(train_set.user_map),
                            n_items=len(train_set.item_map),
                            features_dim=items_features.shape[1],
                            gamma_dim=self.gamma_dim,
                            theta_dim=self.theta_dim,
                            device=self.device)

        optimizer = self.train_optimizer([
            model.beta_items,
            model.gamma_users,
            model.gamma_items,
            model.theta_users,
            model.E,
            model.beta_prime
        ], **self.train_optimizer_parameters)

        train_dataset = TriplesDataset(train_set, self.seed)

        train_dl = torch.utils.data.DataLoader(train_dataset, **self.dl_parameters)

        model.train()

        logger.info("Starting VBPR training!")
        for epoch in range(self.epochs):

            train_loss = 0
            n_user_processed = 0

            with get_progbar(train_dl) as pbar:

                pbar.set_description(f"Starting {epoch + 1}/{self.epochs} epoch...")

                for i, batch in enumerate(pbar):

                    user_idx = batch[0].long()
                    pos_idx = batch[1].long()
                    neg_idx = batch[2].long()

                    n_user_processed += len(user_idx)

                    positive_features = items_features[pos_idx]
                    negative_features = items_features[neg_idx]

                    model_input = (
                        user_idx.to(self.device),
                        pos_idx.to(self.device),
                        neg_idx.to(self.device),
                        positive_features.to(self.device),
                        negative_features.to(self.device)
                    )

                    Xuij, (gamma_u, theta_u), (beta_i_pos, beta_i_neg), (gamma_i_pos, gamma_i_neg) = model(model_input)
                    loss = - self.train_loss(Xuij).sum()

                    reg = (
                            _l2_loss(gamma_u, gamma_i_pos, gamma_i_neg, theta_u) * self.lambda_w
                            + _l2_loss(beta_i_pos) * self.lambda_b_pos
                            + _l2_loss(beta_i_neg) * self.lambda_b_neg
                            + _l2_loss(model.E, model.beta_prime) * self.lambda_e
                    )

                    loss = loss + reg
                    train_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i + 1) % 100 == 0 or (i + 1) == len(train_dl):
                        pbar.set_description(f'[Epoch {epoch + 1}/{self.epochs}, '
                                             f'Batch {i + 1}/{len(train_dl)}, '
                                             f'Loss: {train_loss / n_user_processed:.3f}]')

        logger.info("Training complete!")

        logger.info("Computing visual bias and theta items for faster ranking...")
        with torch.no_grad():
            model.theta_items = items_features.mm(model.E.data).cpu()
            model.visual_bias = items_features.mm(model.beta_prime.data).squeeze().cpu()
            model.cpu()

        logger.info("Done!")

        return model

    def rank(self, fit_alg: VBPRNetwork, train_set: Ratings, test_set: Ratings, items_directory: str,
             user_idx_list: Set[int], n_recs: Optional[int], methodology: Methodology,
             num_cpus: int) -> List[np.ndarray]:

        def compute_single_rank(user_idx):
            filter_list = methodology.filter_single(user_idx, train_set, test_set)
            user_rank = fit_alg.return_scores(user_idx, filter_list).cpu().numpy()
            user_uir = np.array((
                np.full(len(user_rank), user_idx),
                filter_list,
                user_rank
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

    def predict(self, fit_alg: VBPRNetwork, train_set: Ratings, test_set: Ratings, items_directory: str,
                user_idx_list: Set[int], methodology: Methodology,
                num_cpus: int) -> List[np.ndarray]:

        raise NotPredictionAlg("VBPR is not a Score Prediction Algorithm!")

    def fit_rank(self, train_set: Ratings, test_set: Ratings, items_directory: str, user_idx_list: Set[int],
                 n_recs: Optional[int], methodology: Methodology,
                 num_cpus: int, save_fit: bool) -> Tuple[Optional[VBPRNetwork], List[np.ndarray]]:

        vbpr_fit = self.fit(train_set, items_directory, num_cpus)
        rank = self.rank(vbpr_fit, train_set, test_set, items_directory, user_idx_list, n_recs, methodology, num_cpus)

        vbpr_fit = vbpr_fit if save_fit else None

        return vbpr_fit, rank

    def fit_predict(self, train_set: Ratings, test_set: Ratings, items_directory: str, user_idx_list: Set[int],
                    methodology: Methodology,
                    num_cpus: int, save_fit: bool) -> Tuple[Optional[VBPRNetwork], List[np.ndarray]]:

        raise NotPredictionAlg("VBPR is not a Score Prediction Algorithm!")

    def __str__(self):
        return "VBPR"

    def __repr__(self):
        return f"VBPR(item_field={self.item_field}, gamma_dim={self.gamma_dim}, theta_dim={self.theta_dim}, " \
               f"batch_size={self.dl_parameters['batch_size']}, epochs={self.epochs}, threshold={self.threshold}, " \
               f"additional_opt_parameters={self.train_optimizer_parameters['lr']}, lambda_w={self.lambda_w}, " \
               f"lambda_b_pos={self.lambda_b_pos}, lambda_b_neg={self.lambda_b_neg}, lambda_e={self.lambda_e}, " \
               f"train_loss={repr(self.train_loss)}, optimizer_class={repr(self.train_optimizer)}, " \
               f"device={self.device}, embedding_combiner={repr(self._embedding_combiner)}, " \
               f"normalize={self.normalize}, seed={self.seed}, " \
               f"additional_opt_parameters={ {key: val for key, val in self.dl_parameters.items() if key != 'lr'} }, " \
               f"additional_dl_parameters={ {key: val for key, val in self.dl_parameters.items() if key != 'batch_size'} })"
