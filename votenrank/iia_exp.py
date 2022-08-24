import pandas as pd
import numpy as np

from tqdm.auto import tqdm, trange

from . import Leaderboard


def fine_sorted_ranking(ranking):
    big_list = [(rank, model) for model, rank in ranking.iteritems()]
    big_list.sort(reverse=True)
    return [model for rank, model in big_list]


def compute_iia_for_fixed_models(method, table, models_order, weights):
    result = 0

    base_lb = Leaderboard(table.loc[models_order[:2]], weights)
    last_ranking = fine_sorted_ranking(eval(f"base_lb.{method}_ranking()"))

    for current_models_order in range(3, len(models_order)):
        current_lb = Leaderboard(table.loc[models_order[:current_models_order]], weights)
        current_ranking = fine_sorted_ranking(eval(f"current_lb.{method}_ranking()"))
        current_ranking_without_new_model = current_ranking.copy()
        current_ranking_without_new_model.remove(models_order[current_models_order - 1])
        result += last_ranking != current_ranking_without_new_model
        last_ranking = current_ranking
    
    return result


def compute_iia(method, table, weights, num_repetitions):
    results = []
    for i in trange(num_repetitions, leave=False):
        models_order = table.index.tolist()
        np.random.seed(i)
        np.random.shuffle(models_order)
        results.append(compute_iia_for_fixed_models(method, table, models_order, weights))
    return np.mean(results), np.std(results), results
