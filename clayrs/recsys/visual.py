import functools
import itertools
import gc
import os
from typing import Any, Set, Optional, List, Type, Dict, Callable

from clayrs.content_analyzer import Ratings, Centroid
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique
from clayrs.content_analyzer.ratings_manager.ratings import Interaction
from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from clayrs.recsys.methodology import Methodology, TestRatingsMethodology

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.utils.data as data

from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_iterator_parallel, get_progbar


class TriplesDataset(data.Dataset):
    def __init__(self, train_ratings: Ratings,
                 user_map: Dict[str, int],
                 item_map: Dict[str, int],
                 features: np.ndarray,
                 seed: int):
        self.train_ratings = train_ratings
        self.user_map = user_map
        self.item_map = item_map
        self.features = features
        self.n_items = len(item_map)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.train_ratings)

    @functools.lru_cache(maxsize=128)
    def user_positive_interactions(self, user_id) -> Set:
        return set(self.item_map[interaction.item_id]
                   for interaction in self.train_ratings.get_user_interactions(user_id))
    
    def __getitem__(self, idx):
        user_id: str = self.train_ratings.user_id_column[idx]
        user_idx: int = self.user_map[user_id]

        pos_item_id: str = self.train_ratings.item_id_column[idx]
        pos_item_idx: int = self.item_map[pos_item_id]

        neg_item_idx: int = self.rng.choice(self.n_items)
        while neg_item_idx in self.user_positive_interactions(user_id):
            neg_item_idx = self.rng.choice(self.n_items)

        return user_idx, pos_item_idx, neg_item_idx, self.features[pos_item_idx], self.features[neg_item_idx]


class VBPRNetwork(torch.nn.Module):

    def __init__(self, n_users, n_items, features_dim, gamma_dim, theta_dim, seed=None):

        super().__init__()

        self.seed = seed

        self.gamma_users = nn.Embedding(n_users, gamma_dim)
        self.gamma_items = nn.Embedding(n_items, gamma_dim)

        self.theta_users = nn.Embedding(n_users, theta_dim)
        self.E = nn.Embedding(features_dim, theta_dim)

        self.beta_items = nn.Embedding(n_items, 1)
        self.beta_prime = nn.Embedding(features_dim, 1)

        self._init_weights()

        self.theta_items: Optional[torch.Tensor] = None
        self.visual_bias: Optional[torch.Tensor] = None

    def _init_weights(self):

        if self.seed:
            torch.manual_seed(self.seed)

        nn.init.zeros_(self.beta_items.weight)
        nn.init.xavier_uniform_(self.gamma_users.weight)
        nn.init.xavier_uniform_(self.gamma_items.weight)
        nn.init.xavier_uniform_(self.theta_users.weight)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.beta_prime.weight)

    def forward(self, x):
        users = x[0]
        pos_items = x[1]
        neg_items = x[2]
        pos_items_features = x[3]
        neg_items_features = x[4]

        feature_diff = pos_items_features - neg_items_features

        beta_items_pos = self.beta_items(pos_items)
        beta_items_neg = self.beta_items(neg_items)
        beta_items_diff = beta_items_pos - beta_items_neg

        user_gamma = self.gamma_users(users)
        user_theta = self.theta_users(users)

        gamma_items_pos = self.gamma_items(pos_items)
        gamma_items_neg = self.gamma_items(neg_items)
        gamma_items_diff = gamma_items_pos - gamma_items_neg

        theta_item_diff = torch.mm(feature_diff, self.E.weight)

        # this is the same of doing inner products!
        Xuij = (
                beta_items_diff.squeeze() +
                (user_gamma * gamma_items_diff).sum(dim=1) +
                (user_theta * theta_item_diff).sum(dim=1) +
                torch.mm(feature_diff.float(), self.beta_prime.weight)
        )

        return Xuij, (user_gamma, user_theta), (beta_items_pos, beta_items_neg), (gamma_items_pos, gamma_items_neg)

    def return_scores(self, user_idx, item_idx=None):

        if item_idx is not None:
            items_idx_tensor = torch.tensor(item_idx).to("cuda:0").long()
            beta_items = self.beta_items(items_idx_tensor)
            theta_items = self.theta_items[items_idx_tensor]
            gamma_items = self.gamma_items(items_idx_tensor)
            visual_bias = self.visual_bias[items_idx_tensor]
        else:
            beta_items = self.beta_items.weight
            gamma_items = self.gamma_items.weight
            theta_items = self.theta_items
            visual_bias = self.visual_bias

        user_idx_tensor = torch.tensor(user_idx).to("cuda:0")

        with torch.no_grad():
            x_u = (
                    beta_items.squeeze() +
                    visual_bias.squeeze() +
                    torch.matmul(gamma_items, self.gamma_users(user_idx_tensor)) +
                    torch.matmul(theta_items, self.theta_users(user_idx_tensor))
            )

        return x_u.cpu().numpy()


class VBPR(ContentBasedAlgorithm):

    def __init__(self, item_field: dict,
                 gamma_dim: int, theta_dim: int, batch_size: int, epochs: int,
                 learning_rate: float = 0.005,
                 lambda_w: float = 0.01, lambda_b_pos: float = 0.01, lambda_b_neg: float = 0.001, lambda_e: float = 0,
                 train_loss: Callable[[torch.Tensor], torch.Tensor] = fun.logsigmoid,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 device: str = None,
                 embedding_combiner: CombiningTechnique = Centroid(),
                 normalize: bool = True,
                 seed: int = None,
                 additional_opt_parameters: Dict[str, Any] = None,
                 additional_dl_parameters: Dict[str, Any] = None,
                 user_map: Dict[str, int] = None,
                 item_map: Dict[str, int] = None):
        super().__init__(item_field, None)

        if additional_opt_parameters is None:
            additional_opt_parameters = {}

        if additional_dl_parameters is None:
            additional_dl_parameters = {}

        additional_opt_parameters["lr"] = learning_rate
        additional_dl_parameters["batch_size"] = batch_size
        additional_dl_parameters["shuffle"] = False

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

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

        # mappers, they are used to map each item id (or user id) to its corresponding tensor in the VBPR parameters
        # for example, an item with index 0 will refer to the first feature vector in the features embedding
        self.user_id_to_idx_map = user_map if user_map is not None else {}
        self.item_id_to_idx_map = item_map if item_map is not None else {}

    def fit(self, train_set: Ratings, items_directory: str, num_cpus: int = 0):

        if len(self.user_id_to_idx_map) == 0:
            all_users = set(train_set.user_id_column)
            for user in all_users:
                self.user_id_to_idx_map[user] = len(self.user_id_to_idx_map)

        if len(self.item_id_to_idx_map) == 0:
            all_items = set(filename.split(".")[0] for filename in os.listdir(items_directory))
            for item_id in all_items:
                self.item_id_to_idx_map[item_id] = len(self.item_id_to_idx_map)

        loaded_items_interface = self._load_available_contents(items_directory, set())

        logger.info("Loading items features")

        items_features = []

        with get_progbar(self.item_id_to_idx_map) as pbar:
            pbar.set_description("Loading features from serialized items...")

            for item_id in pbar:

                item = loaded_items_interface.get(item_id, self.item_field, throw_away=True)

                if item is not None:
                    item_features = self.fuse_representations([self.extract_features_item(item)],
                                                              self._embedding_combiner)

                    items_features.append(item_features)
                else:
                    items_features.append(None)

        # temporary(?) solution for items that appear in the ratings but are not stored locally
        if len(items_features) == 0:
            raise Exception("No items were loaded")

        # there's surely at least one element
        for i, item_feature in enumerate(items_features):
            if item_feature is None:
                first_not_none_element = next(item for item in items_features if item is not None)
                items_features[i] = np.zeros(shape=first_not_none_element.shape)

        items_features = torch.from_numpy(np.vstack(items_features)).float()

        if self.normalize is True:
            items_features = items_features - torch.min(items_features)
            items_features = items_features / (torch.max(items_features) + 1e-10)

        del loaded_items_interface
        gc.collect()

        torch.cuda.empty_cache()

        model = VBPRNetwork(n_users=len(self.user_id_to_idx_map),
                            n_items=len(self.item_id_to_idx_map),
                            features_dim=items_features.shape[1],
                            gamma_dim=self.gamma_dim,
                            theta_dim=self.theta_dim,
                            seed=self.seed).float().to(self.device)
        optimizer = self.train_optimizer(model.parameters(), **self.train_optimizer_parameters)

        train_dataset = TriplesDataset(train_set, self.user_id_to_idx_map,
                                       self.item_id_to_idx_map, items_features, self.seed)

        train_dl = torch.utils.data.DataLoader(train_dataset, **self.dl_parameters)

        logger.info("STARTING VBPR TRAINING")

        def _l2_loss(*tensors):
            l2_loss = 0
            for tensor in tensors:
                l2_loss += tensor.pow(2).sum()
            return l2_loss / 2

        for epoch in range(self.epochs):

            train_loss = 0
            n_user_processed = 0

            with get_progbar(train_dl) as pbar:
                pbar.set_description("Performing training...")

                for i, batch in enumerate(pbar):

                    n_user_processed += len(batch[0])

                    batch[0] = batch[0].to(self.device)
                    batch[1] = batch[1].to(self.device)
                    batch[2] = batch[2].to(self.device)
                    batch[3] = batch[3].float().to(self.device)
                    batch[4] = batch[4].float().to(self.device)

                    Xuij, (gamma_u, theta_u), (beta_i_pos, beta_i_neg), (gamma_i_pos, gamma_i_neg) = model(batch)
                    loss = - self.train_loss(Xuij).sum()

                    reg = (
                            _l2_loss(gamma_u, gamma_i_pos, gamma_i_neg, theta_u) * self.lambda_w
                            + _l2_loss(beta_i_pos) * self.lambda_b_pos
                            + _l2_loss(beta_i_neg) * self.lambda_b_neg
                            + _l2_loss(model.E.weight, model.beta_prime.weight) * self.lambda_e
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

        model.eval()

        # features_from_item_map = dict(sorted({item_map[id]: f for id, f in zip(item_map.keys(), features)}.items()))
        model.visual_bias = torch.mm(items_features, model.beta_prime.weight.data.cpu()).to("cuda:0")
        model.theta_items = torch.mm(items_features, model.E.weight.data.cpu()).to("cuda:0")
        return model

    def rank(self, fit_alg: VBPRNetwork, train_set: Ratings, test_set: Ratings, items_directory: str, user_id_list: Set,
             n_recs: Optional[int] = None, methodology: Optional[Methodology] = TestRatingsMethodology(),
             num_cpus: int = 1) -> List[Interaction]:

        def compute_single_rank(user_id):
            user_id = str(user_id)
            user_ratings = train_set.get_user_interactions(user_id)

            filter_list = None
            if methodology is not None:
                filter_list = set(methodology.filter_single(user_id, train_set, test_set))

            # Load items to predict
            if filter_list is None:
                user_seen_items = set([interaction.item_id for interaction in user_ratings])
                items_id_to_predict = list(set(train_set.item_id_column).difference(user_seen_items))
            else:
                items_id_to_predict = list(filter_list)

            items_idx_to_predict = [self.item_id_to_idx_map[item_id] for item_id in items_id_to_predict]

            if len(items_idx_to_predict) > 0:
                user_rank = fit_alg.return_scores(self.user_id_to_idx_map[user_id], items_idx_to_predict)
            else:
                user_rank = []

            user_rank = [Interaction(user_id, item_id, score)
                         for item_id, score in zip(items_idx_to_predict, user_rank)]

            return user_id, user_rank

        rank = []

        with get_iterator_parallel(num_cpus,
                                   compute_single_rank, user_id_list,
                                   progress_bar=True, total=len(user_id_list)) as pbar:

            for user_id, user_rank in pbar:
                pbar.set_description(f"Computing rank for user {user_id}")
                rank.append(user_rank)

        return list(itertools.chain.from_iterable(rank))

    def predict(self, fit_alg: Any, train_set: Ratings, test_set: Ratings, items_directory: str, user_id_list: Set,
                n_recs: Optional[int] = None, methodology: Optional[Methodology] = TestRatingsMethodology(),
                num_cpus: int = 1) -> List[Interaction]:
        pass
