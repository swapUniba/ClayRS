from __future__ import annotations
import functools
from typing import Set, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

if TYPE_CHECKING:
    from clayrs.content_analyzer import Ratings


class TriplesDataset(data.Dataset):
    def __init__(self, train_ratings: Ratings, seed: int):

        self.train_ratings = train_ratings
        self.n_items = len(train_ratings.item_map)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.train_ratings)

    @functools.lru_cache(maxsize=128)
    def user_positive_interactions(self, user_idx: int) -> Set:
        return set(self.train_ratings.item_idx_column[self.train_ratings.get_user_interactions(user_idx,
                                                                                               as_indices=True)])

    def __getitem__(self, idx):

        user_idx = self.train_ratings.user_idx_column[idx]
        pos_item_idx = self.train_ratings.item_idx_column[idx]

        neg_item_idx: int = self.rng.choice(self.n_items)
        while neg_item_idx in self.user_positive_interactions(user_idx):
            neg_item_idx = self.rng.choice(self.n_items)

        return user_idx, pos_item_idx, neg_item_idx


class VBPRNetwork(torch.nn.Module):

    def __init__(self, n_users, n_items, features_dim, gamma_dim, theta_dim, device):

        super().__init__()

        self.device = device

        self.beta_items = nn.Parameter(torch.zeros(size=(n_items,), dtype=torch.float), requires_grad=True)
        self.gamma_users = nn.Parameter(torch.zeros(size=(n_users, gamma_dim), dtype=torch.float), requires_grad=True)
        self.gamma_items = nn.Parameter(torch.zeros(size=(n_items, gamma_dim), dtype=torch.float), requires_grad=True)

        self.theta_users = nn.Parameter(torch.zeros(size=(n_users, theta_dim), dtype=torch.float), requires_grad=True)
        self.E = nn.Parameter(torch.zeros(size=(features_dim, theta_dim), dtype=torch.float), requires_grad=True)

        self.beta_prime = nn.Parameter(torch.zeros(size=(features_dim, 1), dtype=torch.float), requires_grad=True)

        self._init_weights()

        self.to(device)

        self.theta_items: Optional[torch.Tensor] = None
        self.visual_bias: Optional[torch.Tensor] = None

    def _init_weights(self):
        nn.init.zeros_(self.beta_items)
        nn.init.xavier_uniform_(self.gamma_users)
        nn.init.xavier_uniform_(self.gamma_items)
        nn.init.xavier_uniform_(self.theta_users)
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.beta_prime)

    def forward(self, x):
        users = x[0]
        pos_items = x[1]
        neg_items = x[2]
        pos_items_features = x[3]
        neg_items_features = x[4]

        feature_diff = pos_items_features - neg_items_features

        beta_items_pos = self.beta_items[pos_items]
        beta_items_neg = self.beta_items[neg_items]
        beta_items_diff = beta_items_pos - beta_items_neg

        user_gamma = self.gamma_users[users]
        user_theta = self.theta_users[users]

        gamma_items_pos = self.gamma_items[pos_items]
        gamma_items_neg = self.gamma_items[neg_items]
        gamma_items_diff = gamma_items_pos - gamma_items_neg

        theta_item_diff = feature_diff.mm(self.E)

        # this is the same of doing inner products!
        Xuij = (
                beta_items_diff +
                (user_gamma * gamma_items_diff).sum(dim=1) +
                (user_theta * theta_item_diff).sum(dim=1) +
                feature_diff.mm(self.beta_prime)
        )

        return Xuij, (user_gamma, user_theta), (beta_items_pos, beta_items_neg), (gamma_items_pos, gamma_items_neg)

    def return_scores(self, user_idx, item_idx):

        with torch.no_grad():
            items_idx_tensor = torch.tensor(item_idx).long()
            beta_items = self.beta_items[items_idx_tensor]
            theta_items = self.theta_items[items_idx_tensor]
            gamma_items = self.gamma_items[items_idx_tensor]
            visual_bias = self.visual_bias[items_idx_tensor]

            # in case a score must be returned for each item fitted, set `item_idx = None`
            # and appropriately check this case. The below is the implementation in case `item_idx is None`
            #
            #     beta_items = self.beta_items.weight
            #     gamma_items = self.gamma_items.weight
            #     theta_items = self.theta_items
            #     visual_bias = self.visual_bias

            user_idx_tensor = torch.tensor(user_idx)

            x_u = (
                    beta_items +
                    visual_bias +
                    torch.matmul(gamma_items, self.gamma_users[user_idx_tensor]) +
                    torch.matmul(theta_items, self.theta_users[user_idx_tensor])
            )

        return x_u.numpy()
