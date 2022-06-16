import scipy.stats as stats
import os
import pandas as pd
from tqdm.notebook import tqdm

from experiment_impact_tracker.data_interface import DataInterface
from experiment_impact_tracker.data_utils import load_initial_info
from experiment_impact_tracker.utils import gather_additional_info


def ranking2top(ranking):
    return ranking[ranking == ranking.max()].index.tolist()


def kendall_tau(df):
    res_d = {}
    for method, subset in df.iteritems():
        res_d[method] = subset.apply(lambda x: x.split(":")[1].strip()).tolist()

    return {
        method: round(stats.kendalltau(res_d["AM"], method_top_k)[0], 3)
        for method, method_top_k in res_d.items()
        if method != "AM"
    }


def agreement_rate(df, k, top_k=True):
    res_d = {}
    for method, subset in df.iteritems():
        _subset = subset.copy().iloc[:k] if top_k else subset.copy().iloc[-k:]
        res_d[method] = _subset.apply(lambda x: x.split(":")[1].strip()).tolist()

    return {
        method: round(len(set(method_top_k).intersection(set(res_d["AM"]))) / k, 2)
        for method, method_top_k in res_d.items()
        if method != "AM"
    }


def tracker_filename(model, task, dirpath):
    return f"{dirpath}/{model}_{task}_0/"


def get_tracker_table(data, dirpath):
    di_attrs = ["exp_len_hours", "kg_carbon", "total_power"]
    info_attrs = ["gpu_hours"]

    tracker_cols = []
    for task in data.columns:
        tracker_cols += [
            task.split(".")[0] + "_" + attr for attr in di_attrs + info_attrs
        ]

    tracker_cols = list(set(tracker_cols))
    tracker_res = pd.DataFrame(columns=tracker_cols, index=data.index)

    for f in tqdm(os.listdir(dirpath)):
        model, task, _ = f.split("_")
        if model not in data.index:
            continue

        fname = tracker_filename(model, task, dirpath)
        datain = DataInterface([fname + "impacttracker"])
        info = load_initial_info(fname)
        add_info = gather_additional_info(info, fname)

        for attr in di_attrs:
            tracker_res.loc[model, f"{task}_{attr}"] = getattr(datain, attr)
        for attr in info_attrs:
            tracker_res.loc[model, f"{task}_{attr}"] = add_info[attr]

    return tracker_res
