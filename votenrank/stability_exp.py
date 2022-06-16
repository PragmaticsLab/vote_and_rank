from scipy.stats import spearmanr
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

from . import Leaderboard


def spearman_exp(lb, num_repeats, exp_range, top_k=7):
    res_corrs = []

    with tqdm(total=num_repeats * len(exp_range)) as pbar:
        for repeat_idxs in range(num_repeats):
            ranks = []
            tables = []

            for nan_number in exp_range:
                nan_number = int(round(nan_number * lb.table.count().sum()))
                table_nan = lb.table.copy()

                nan_idxs_prod = np.random.choice(
                    table_nan.shape[0] * table_nan.shape[1],
                    size=nan_number,
                    replace=False,
                )
                for idx in nan_idxs_prod:
                    table_nan.iloc[
                        idx % table_nan.shape[0], idx // table_nan.shape[0]
                    ] = np.nan

                noised_lb = Leaderboard(table_nan, weights=lb.weights.to_dict())
                tables.append(table_nan)
                ranks.append(
                    noised_lb.rank_all(
                        use_methods={
                            "mean": {"mean_type": ["arithmetic"]},
                            "minimax": {"score_type": ["winning_votes"]},
                            "copeland": {"slice_type": ["difference"]},
                        }
                    )
                )
                pbar.update(1)

            etalon = {}

            for col in ranks[0].columns:
                scores = ranks[0][col].apply(lambda x: float(x.split(":")[0]))
                methods = ranks[0][col].apply(lambda x: x.split(":")[1].strip())

                norm_rank = pd.Series(index=methods.values, data=scores.values)

                etalon[col] = norm_rank.reindex(tables[0].index)

            corrs = defaultdict(lambda: [])

            for cur_rankings in ranks:
                for col in cur_rankings.columns:
                    scores = cur_rankings[col].apply(lambda x: float(x.split(":")[0]))
                    methods = cur_rankings[col].apply(lambda x: x.split(":")[1].strip())

                    topk_methods = methods.values[:top_k]
                    norm_rank = pd.Series(
                        index=methods.values, data=scores.values
                    ).reindex(tables[0].index)

                    corrs[col].append(
                        spearmanr(
                            etalon[col].loc[topk_methods], norm_rank.loc[topk_methods]
                        )[0]
                    )

            res_corrs.append(corrs)

    final_corrs = defaultdict(int)
    for corr in res_corrs:
        for col, vals in corr.items():
            if len(vals) > 1:
                final_corrs[col] = np.array(vals) + final_corrs[col]

    for col in final_corrs:
        final_corrs[col] /= num_repeats

    return final_corrs


def count_and_plot(lb, num_repeats, exp_range, top_k=7):
    exp_res = spearman_exp(lb, num_repeats, exp_range, top_k)
    for col, nums in exp_res.items():
        sns.lineplot(x=exp_range, y=nums, label=col)


def get_res_df(exp_range, exp_res):
    dfs = []
    for name, nums in exp_res.items():
        df = pd.DataFrame()
        df["criterion"] = exp_range
        df["correlation"] = nums
        df["method"] = name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def create_exp_pic(exp_range, exp_res, filename=None):
    df = get_res_df(exp_range * 100, exp_res)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.lineplot(
        x="criterion",
        y="correlation",
        hue="method",
        style="method",
        data=df,
        linewidth=5,
        ci=100,
    )

    plt.yticks(fontsize=20)
    plt.xticks(range(0, 21, 5), fontsize=20)
    plt.ylabel("$\\rho$", fontsize=30)
    plt.xlabel("Missing values (%)", fontsize=25)
    plt.grid()
    plt.tight_layout()

    L = plt.legend(fontsize=20, loc=3)
    for line in L.get_lines():
        line.set_linewidth(5.0)
    L.get_texts()[0].set_text("$\\sigma^{am}$")

    if filename is not None:
        plt.savefig(filename, format="pdf")
