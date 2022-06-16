import numpy as np
from scipy.optimize import linprog

from typing import Dict, List, Tuple


def _get_tasks_onehot(self, tasks: List[str]):
    idxs = [0] * len(self.tasks)
    for task in tasks:
        if task not in self.tasks:
            raise ValueError(f"Invalid task {task}")
        idxs[self.tasks.index(task)] = 1
    return idxs


def _find_weights_for_majority_graph(
    self, edge_list: List[Tuple[str, str]], restrictions: Dict[str, List] = None
):
    params = {
        "weights_to_minimize": None,
        "weights_to_maximize": None,
        "weights_ub": None,
        "weights_lb": None,
        "weights_inequality": None,
    }
    if restrictions is None:
        restrictions = {}

    if len(set(restrictions.keys()) - set(params.keys())) > 0:
        raise ValueError(
            f"Invalid keys {set(restrictions.keys()) - set(params.keys())}"
        )
    params.update(restrictions)

    if params["weights_to_minimize"] is None and params["weights_to_maximize"] is None:
        objective = [1] * self.n_tasks
    elif (
        params["weights_to_minimize"] is not None
        and params["weights_to_maximize"] is not None
    ):
        raise ValueError(
            f"Only one parameter among weights_to_minimize and weights_to_maximize"
            "can be not None"
        )
    elif params["weights_to_minimize"] is not None:
        objective = self._get_tasks_onehot(params["weights_to_minimize"])
    else:
        max_idxs = self._get_tasks_onehot(params["weights_to_maximize"])
        objective = [-el for el in max_idxs]

    weights_cond_left = []
    weights_cond_right = []

    if params["weights_ub"] is not None:
        for weights_list, ub in params["weights_ub"]:
            weights_cond_left.append(self._get_tasks_onehot(weights_list))
            weights_cond_right.append(ub)

    if params["weights_lb"] is not None:
        for weights_list, lb in params["weights_lb"]:
            weights_cond_left.append(
                [-el for el in self._get_tasks_onehot(weights_list)]
            )
            weights_cond_right.append(-lb)

    if params["weights_inequality"] is not None:
        for left_weights, right_weights in params["weights_inequality"]:
            left_ohe = self._get_tasks_onehot(left_weights)
            right_ohe = self._get_tasks_onehot(right_weights)
            cond_coefs = (np.array(left_ohe) - np.array(right_ohe)).tolist()
            weights_cond_left.append(cond_coefs)
            weights_cond_right.append(0)

    left_ineqs = weights_cond_left
    right_ineqs = weights_cond_right

    for looser_model, winner_model in edge_list:
        losses = (self.table.loc[looser_model] > self.table.loc[winner_model]).replace(
            {True: 1, False: 0}
        )
        wins = (self.table.loc[winner_model] > self.table.loc[looser_model]).replace(
            {True: -1, False: 0}
        )
        current_ineq = losses + wins
        left_ineqs.append(current_ineq.tolist())
        right_ineqs.append(0)

    left_eq = [[1] * self.n_tasks]
    right_eq = [1]

    bnd = [(0, 1)] * self.n_tasks

    sol = linprog(
        c=objective,
        A_ub=left_ineqs,
        b_ub=right_ineqs,
        A_eq=left_eq,
        b_eq=right_eq,
        bounds=bnd,
    )
    if sol["success"]:
        return {task: weight for task, weight in zip(self.tasks, sol["x"])}
    else:
        return "infeasible"
