import pandas as pd
import numpy as np

from scipy.stats import gmean

from ..utils import ranking2top


def mean_ranking(self, mean_type: str = "arithmetic"):
    table = self.table.copy()
    if self.is_partial:
        table = table.fillna(table.median())

    if mean_type == "arithmetic":
        return (
            (table * self.weights / self.weights.sum())
            .sum(axis=1)
            .sort_values(ascending=False)
        )
    elif mean_type == "geometric":
        scores = pd.Series(
            index=table.index, data=gmean(table, axis=1, weights=self.weights)
        )
        return scores.sort_values(ascending=False)
    else:
        raise ValueError(
            f"Only arithmetic and geometric mean is supported, got {mean_type}"
        )


def mean_election(self, mean_type: str = "arithmetic"):
    return ranking2top(self.mean_ranking(mean_type=mean_type))


def _approval_ranking(self, acceptance_threshold: int, rank_type: str = "max"):
    if rank_type == "min":
        return (
            ((self.ranks <= acceptance_threshold) * self.weights)
            .sum(axis=1)
            .sort_values(ascending=False)
        )
    elif rank_type == "max":
        return (
            ((self.max_ranks <= acceptance_threshold) * self.weights)
            .sum(axis=1)
            .sort_values(ascending=False)
        )
    else:
        raise ValueError("Rank type should be min or max")


def plurality_ranking(self):
    return self._approval_ranking(1)


def plurality_election(self):
    return ranking2top(self.plurality_ranking())


def threshold_election(self):
    candidate_models = self.models
    for step in range(self.n_models, 1, -1):
        current_ranking = (
            (self.max_ranks.loc[candidate_models] != step) * self.weights
        ).sum(axis=1)
        candidate_models = ranking2top(current_ranking)

    return candidate_models


def borda_ranking(self):
    return (
        ((self.n_models - self.max_ranks) * self.weights)
        .sum(axis=1)
        .sort_values(ascending=False)
    )


def borda_election(self):
    return ranking2top(self.borda_ranking())


def dowdall_ranking(self):
    return ((1 / self.ranks) * self.weights).sum(axis=1).sort_values(ascending=False)


def dowdall_election(self):
    return ranking2top(self.dowdall_ranking())


def condorcet_election(self):
    return self.majority_graph.index[(self.majority_graph == 1).all(axis=1)].tolist()


def baldwin_election(self):
    current_borda = self.borda_ranking()
    while current_borda.min() != current_borda.max():
        candidates = current_borda.index[current_borda != current_borda.min()]
        current_max_ranks = (
            self.table.loc[candidates].rank(method="max", ascending=False).astype(int)
        )
        weighted_ranks = (len(candidates) - current_max_ranks) * self.weights
        current_borda = weighted_ranks.sum(axis=1).sort_values(ascending=False)

    return current_borda.index.tolist()


def copeland_ranking(self, slice_type: str = "lower_with_ties"):
    if slice_type == "lower_with_ties":
        return (self.majority_graph.sum(axis=1) - 1).sort_values(ascending=False)
    elif slice_type == "difference":
        lower_ranking = (self.majority_graph == 1).sum(axis=1) - 1
        upper_ranking = (self.majority_graph == 0).sum(axis=1)
        return (lower_ranking - upper_ranking).sort_values(ascending=False)
    elif slice_type == "lower":
        return ((self.majority_graph == 1).sum(axis=1) - 1).sort_values(ascending=False)
    elif slice_type == "upper":
        return (-(self.majority_graph == 0).sum(axis=1)).sort_values(ascending=False)
    else:
        raise ValueError(
            "Slice type should be lower_with_ties, difference, lower or upper"
        )


def copeland_election(self, slice_type: str = "lower_with_ties"):
    return ranking2top(self.copeland_ranking(slice_type))


def minimax_ranking(self, score_type: str = "winning_votes"):
    ranks = []
    for model in self.models:
        if score_type == "winning_votes":
            models_scores = ((self.ranks < self.ranks.loc[model]) * self.weights).sum(
                axis=1
            )
            does_win = ((self.ranks < self.ranks.loc[model]) * self.weights).sum(
                axis=1
            ) > ((self.ranks > self.ranks.loc[model]) * self.weights).sum(axis=1)
            models_scores *= does_win
        elif score_type == "margins":
            models_scores = ((self.ranks < self.ranks.loc[model]) * self.weights).sum(
                axis=1
            ) - ((self.ranks.loc[model] < self.ranks) * self.weights).sum(axis=1)
        elif score_type == "pairwise_opposition":
            models_scores = ((self.ranks < self.ranks.loc[model]) * self.weights).sum(
                axis=1
            )
        else:
            raise ValueError(
                "Score type should be winning_votes, margins or pairwise_opposition"
            )

        score = models_scores.drop(model).max()
        ranks.append(score)

    return (
        -pd.Series(data=ranks, index=pd.Series(self.models, name="Name"))
    ).sort_values(ascending=False)


def minimax_election(self, score_type: str = "winning_votes"):
    return ranking2top(self.minimax_ranking(score_type))


def optimality_gap_ranking(self, gamma: int):
    gap_scores_np = np.minimum(self.table, gamma) - gamma
    gap_scores = pd.DataFrame(index=self.table.index, columns=self.table.columns, data=gap_scores_np)
    return gap_scores.mean(axis=1).sort_values(ascending=False)


def optimality_gap_election(self, gamma: int):
    return ranking2top(self.optimality_gap_ranking(gamma))
