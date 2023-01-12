import itertools
import gc
from typing import Any, Set, Optional, List, Type, Dict, Callable

from clayrs.content_analyzer import Ratings, Centroid
from clayrs.content_analyzer.field_content_production_techniques.embedding_technique.combining_technique import \
    CombiningTechnique
from clayrs.content_analyzer.ratings_manager.ratings import Interaction
from clayrs.recsys.content_based_algorithm.content_based_algorithm import ContentBasedAlgorithm
from clayrs.recsys.content_based_algorithm.exceptions import EmptyUserRatings
from clayrs.recsys.methodology import Methodology, TestRatingsMethodology

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.utils.data as data

from clayrs.utils.const import logger
from clayrs.utils.context_managers import get_iterator_parallel


class TriplesDataset(data.Dataset):

    def __init__(self, ratings: Ratings,
                 user_map: Dict[str, int], item_map: Dict[str, int], item_repr_map: Dict[int, np.array]):
        self.ratings = ratings
        self.n_items = len(item_repr_map)

        self.user_map = user_map
        self.item_map = item_map
        self.item_repr_map = item_repr_map

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user = self.ratings.user_id_column[idx]
        pos_item = self.item_map[self.ratings.item_id_column[idx]]
        pos_items = [self.item_map[content.item_id] for content in self.ratings.get_user_interactions(user)]
        neg_item = np.random.choice(self.n_items)
        while neg_item in pos_items:
            neg_item = np.random.choice(self.n_items)
        return self.user_map[user], \
               pos_item, \
               neg_item, \
               torch.from_numpy(self.item_repr_map[pos_item]).squeeze().float(), \
               torch.from_numpy(self.item_repr_map[neg_item]).squeeze().float(),


class VBPRNetwork(torch.nn.Module):

    def __init__(self, n_users, n_items, features_dim, gamma_dim, theta_dim):

        super().__init__()

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
        beta_diff = self.beta_items(pos_items) - self.beta_items(neg_items)

        user_gamma = self.gamma_users(users)
        user_theta = self.theta_users(users)

        gamma_item_diff = self.gamma_items(pos_items) - self.gamma_items(neg_items)
        theta_item_diff = torch.mm(feature_diff, self.E.weight)

        x_uij = (
                beta_diff +
                (user_gamma * gamma_item_diff).sum(dim=1) +
                (user_theta * theta_item_diff).sum(dim=1) +
                (torch.mm(feature_diff, self.beta_prime.weight))
        )

        return x_uij

    def get_regularizer(self, batch, lambda_w, lambda_b, lambda_e):
        pass

    def return_scores(self, user_idx, item_idx):

        gamma_u = self.gamma_users(user_idx)
        theta_u = self.theta_users(user_idx)

        gamma_i = self.gamma_items(item_idx)

        x_ui = (
                self.beta_items(item_idx)
                + (gamma_u * gamma_i).sum(dim=1)
                + (theta_u * self.theta_items[item_idx]).sum(dim=1)
                + self.visual_bias[item_idx]
        )

        return x_ui


class VBPR(ContentBasedAlgorithm):

    def __init__(self, item_field: dict,
                 gamma_dim: int, theta_dim: int, train_batch_size: int, train_epochs: int,
                 lambda_w: float = 0.1, lambda_b: float = 0.1, lambda_e: float = 0,
                 train_loss: Callable[[torch.Tensor], torch.Tensor] = fun.logsigmoid,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_parameters: Dict[str, Any] = None,
                 device: str = None,
                 embedding_combiner: CombiningTechnique = Centroid()):
        super().__init__(item_field, None)

        if optimizer_parameters is None:
            optimizer_parameters = {}

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.gamma_dim = gamma_dim
        self.theta_dim = theta_dim

        self.train_batch_size = train_batch_size
        self.train_epochs = train_epochs
        self.train_loss = train_loss
        self.train_optimizer = optimizer_class
        self.train_optimizer_parameters = optimizer_parameters
        self.lambda_w = lambda_w
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e

        self._embedding_combiner = embedding_combiner

        # mappers, they are used to map each item id (or user id) to its corresponding tensor in the VBPR parameters
        # for example, an item with index 0 will refer to the first feature vector in the features embedding
        self.user_id_to_idx_map = {}
        self.item_id_to_idx_map = {}
        self.item_idx_to_id_map = {}

    def fit(self, train_set: Ratings, items_directory: str, num_cpus: int = 0):

        items_to_load = set(train_set.item_id_column)
        all_users = set(train_set.user_id_column)

        for user in all_users:
            self.user_id_to_idx_map[user] = len(self.user_id_to_idx_map)

        loaded_items_interface = self._load_available_contents(items_directory, set())

        logger.info("Loading items features")

        items_features = {}
        missing_items = []

        shape = None

        for i, item_id in enumerate(items_to_load):

            item = loaded_items_interface.get(item_id, self.item_field)

            self.item_id_to_idx_map[item_id] = i
            self.item_idx_to_id_map[i] = item_id

            if item is not None:
                repr = self.fuse_representations([self.extract_features_item(item)], self._embedding_combiner)
                items_features[i] = repr

                if shape is None:
                    shape = repr.shape
            else:
                missing_items.append(item_id)

        # temporary(?) solution for items that appear in the ratings but are not stored locally
        if len(missing_items) > 0:
            if shape is None:
                raise Exception("No items were loaded")
            for item_id in missing_items:
                repr = np.zeros(shape)
                items_features[self.item_id_to_idx_map[item_id]] = repr

        del loaded_items_interface
        gc.collect()

        torch.cuda.empty_cache()

        n_items = len(items_features)

        model = VBPRNetwork(len(all_users), n_items, shape[1], self.gamma_dim, self.theta_dim).to(self.device)
        train_optimizer = self.train_optimizer(model.parameters(), **self.train_optimizer_parameters)

        train_dataset = TriplesDataset(train_set, self.user_id_to_idx_map, self.item_id_to_idx_map, items_features)
        sampler = data.RandomSampler(train_dataset)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_batch_size, sampler=sampler)

        logger.info("STARTING VBPR TRAINING")

        for epoch in range(self.train_epochs):

            train_loss = 0.0
            model.train()

            for i, batch in enumerate(train_dl):

                batch[0] = batch[0].to(self.device)
                batch[1] = batch[1].to(self.device)
                batch[2] = batch[2].to(self.device)
                batch[3] = batch[3].to(self.device)
                batch[4] = batch[4].to(self.device)

                train_optimizer.zero_grad()
                outputs = model(batch)
                loss = - self.train_loss(outputs).sum()
                # loss += model.get_regularizer(batch, self.lambda_w, self.lambda_b, self.lambda_e)
                loss.backward()
                train_loss += loss.item()
                train_optimizer.step()

                if i % 100 == 99:
                    logger.info(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] Loss: {train_loss / 100:.3f}')
                    train_loss = 0.0

        feature_matrix = torch.from_numpy(np.vstack([np.load(feature) for feature in items_features.values()])).float()
        model.theta_items = torch.mm(feature_matrix, model.E.weight.cpu()).to(self.device)
        model.visual_bias = torch.mm(feature_matrix, model.beta_prime.weight.cpu()).to(self.device)
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
            try:
                user_id = user_ratings[0].user_id
            except IndexError:
                raise EmptyUserRatings("The user selected doesn't have any ratings!")

            # Load items to predict
            if filter_list is None:
                user_seen_items = set([interaction.item_id for interaction in user_ratings])
                items_to_predict = [self.item_id_to_idx_map[item_id] for item_id in self.item_id_to_idx_map.keys()
                                    if item_id not in user_seen_items]
            else:
                items_to_predict = [self.item_id_to_idx_map[item_id] for item_id in filter_list]

            items_to_predict = torch.tensor(items_to_predict).to(self.device).long()

            if len(items_to_predict) > 0:
                user_rank = fit_alg.return_scores(
                    torch.tensor(self.user_id_to_idx_map[user_id]).to(self.device), items_to_predict)
            else:
                user_rank = []

            user_rank = [Interaction(user_id, self.item_idx_to_id_map[idx.item()], score)
                         for idx, score in zip(items_to_predict, user_rank)]

            return user_id, user_rank

        rank = []

        logger.info("Don't worry if it looks stuck at first")
        logger.info("First iterations will stabilize the estimated remaining time")

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
