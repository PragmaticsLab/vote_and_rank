## Vote'n'Rank: Revision of Benchmarking with Social Choice Theory

This repository contains the code, experiment results, and other materials used in our [EACL 2023 paper](https://aclanthology.org/2023.eacl-main.48/).


**Authors**: Mark Rofin, Mikhail Florinskiy, Vladislav Mikhailov, Andrey Kravchenko, Elena Tutubalina, Tatiana Shavrina, Daniel Karabekyan, and Ekaterina Artemova

### Abstract
The development of state-of-the-art systems in different applied areas of artificial intelligence (AI) is driven by benchmarks, which have played a crucial role in shaping the paradigm of evaluating generalisation capabilities from multiple perspectives. Although the paradigm is shifting towards more fine-grained evaluation across diverse complex tasks, the delicate question of how to aggregate the performances has received particular interest in the community. The benchmarks generally follow the unspoken *utilitarian* principles, where the systems are ranked based on their mean average score over task-specific metrics. Such aggregation procedure has been viewed as a sub-optimal evaluation protocol, which may have created the illusion of progress in the field. This paper proposes **Vote'n'Rank**, a framework for ranking systems in multi-task benchmarks under the principles of the social choice theory. We demonstrate that our approach can be efficiently utilised to draw new insights on benchmarking in several AI sub-fields and identify the best-performing systems in simulated practical scenarios that meet user needs. The **Vote'n'Rank**'s procedures are empirically shown to be more robust than the mean average while being able to handle missing performance scores and determine conditions under which the system becomes the winner.

### Objective
Our main objective is to re-interpret the common benchmarking trends through the lens of the social choice theory and discuss how the leaderboards get changed if we follow the **Vote'n'Rank** rank aggregation principles. We also analyse the application of our methods to real-life scenarios, where the leaderboard holders do not report the performance for particular tasks, and the user can define the system evaluation criteria based on their end needs.


### Our aggregation procedures
1. **Scoring rules**: Plurality rule, Borda rule, and Dowdall rule.
2. **Iterative scoring rules**: Threshold rule and Baldwin rule.
3. **Majority-relation based rules**: Condorcet rule, Copeland's rule, and Minimax rule.

### Case studies
We consider four case studies (CSs) on three benchmarks across multiple ML fields: GLUE, SuperGLUE, and VALUE. The experiments can be reproduced using the following notebooks:

1. ```CS1 - Reranking Leaderboards.ipynb``` -- re-ranking the leaderboards and selecting the winner.
2. ```CS2 - Condorcet Search.ipynb``` -- identifying prosopective and non-prospective models.
3. ```CS3 - NaN Stability.ipynb``` -- exploring robustness to missing performance scores (replaced with NaNs).
4. ```CS4 - User-Oriented Setting.ipynb``` -- ranking systems in a simulated practical scenarios according to the following criteria: *performance*, *computational efficiency*, and *fairness*.
5. ```IIA.ipynb``` -- experiment on the count-based independence of irrelevant alternatives in Case Study 1.

CS1-CS3 are conducted on the publicly available benchmark leaderboards, while CS4 includes additional experiments run on a single GPU unit, NVIDIA A100 80 GB SXM (NVLink), 4-CPU cores, AMD EPYC 7702 2-3.35 GHz, and 1 TB RAM.

### Data and other materials
The leaderboard results by the access date and CS4 experiment results can be found [here](https://github.com/PragmaticsLab/vote_and_rank/tree/main/tables). 


### Framework quickstart

1. Vote'n'Rank leaderboard can be initialized from a table of models' performance, where each row corresponds to a model and each column corresponds to a task.

    ```
    import pandas as pd
    from votenrank import Leaderboard

    table = pd.DataFrame({"COPA": [70.6, 90.6], "RTE": [71.7, 88.2]}, index=["BERT", "RoBERTa"])
    lb = Leaderboard(table=table, weights={"COPA": 0.6, "RTE": 0.4})
    ```

1. You can rank models using various social choice methods, such as Borda or Copeland.
    ```
    lb.borda_ranking()
    ```

1. The same is true for winner selection.
    ```
    lb.condorcet_election()
    ```

1. To test all methods currently supported in the framework, use `lb.rank_all()` and `lb.elect_all()`.

1. For double-stage ranking, use `two_step_ranking` method.
    ```
    lb.two_step_ranking(
        ranking_method="borda",
        task_groups={"group 1": ["COPA"], "group 2": ["RTE"]}
    )
    ```

1. One can search for prospective models (the models which can become Condorcet winners given particular weights).
    ```
    lb.split_models_by_feasibility()
    ```

### Usage examples
Conda environment.


```
conda install transformers datasets scipy scikit-learn sentencepiece -c conda-forge
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c anaconda cudnn=8.2.1
pip install git+https://github.com/Breakend/experiment-impact-tracker
```

There are a few examples of running the CS4 experiments.
1. Downstream performance evaluation.


```python glue_task.py --model_name="bert-base-uncased" --task_name="wnli" --random_seed=0```

2. Experiment tracker evaluation.


```python glue_task_experiment_tracker.py --model_name="bert-base-uncased" --task_name="wnli" --random_seed=0```

### License
The codebase and experiment materials are available under the Apache 2.0 license. The copyright of the leaderboard results remains with the original authors and benchmark holders.


### Cite us
```
@inproceedings{rofin-etal-2023-votenrank,
    title = "Vote{'}n{'}Rank: Revision of Benchmarking with Social Choice Theory",
    author = "Rofin, Mark  and
      Mikhailov, Vladislav  and
      Florinsky, Mikhail  and
      Kravchenko, Andrey  and
      Shavrina, Tatiana  and
      Tutubalina, Elena  and
      Karabekyan, Daniel  and
      Artemova, Ekaterina",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.48",
    pages = "670--686",
    abstract = "The development of state-of-the-art systems in different applied areas of machine learning (ML) is driven by benchmarks, which have shaped the paradigm of evaluating generalisation capabilities from multiple perspectives. Although the paradigm is shifting towards more fine-grained evaluation across diverse tasks, the delicate question of how to aggregate the performances has received particular interest in the community. In general, benchmarks follow the unspoken utilitarian principles, where the systems are ranked based on their mean average score over task-specific metrics. Such aggregation procedure has been viewed as a sub-optimal evaluation protocol, which may have created the illusion of progress. This paper proposes Vote{'}n{'}Rank, a framework for ranking systems in multi-task benchmarks under the principles of the social choice theory. We demonstrate that our approach can be efficiently utilised to draw new insights on benchmarking in several ML sub-fields and identify the best-performing systems in research and development case studies. The Vote{'}n{'}Rank{'}s procedures are more robust than the mean average while being able to handle missing performance scores and determine conditions under which the system becomes the winner.",
}
```
