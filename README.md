## Vote'n'Rank: Revision of Benchmarking with Social Choice Theory

This repository contains the code, experiment results, and other materials used in our submission to the [NeurIPS 2022 Datasets and Benchmarks Track](https://neurips.cc/Conferences/2022/CallForDatasetsBenchmarks).


**Authors**: Mark Rofin, Mikhail Florinskiy, Vladislav Mikhailov, Andrey Kravchenko, Elena Tutubalina, Tatiana Shavrina, Daniel Karabekyan, and Ekaterina Artemova

### Abstract
The development of state-of-the-art systems in different applied areas of artificial intelligence (AI) is driven by benchmarks, which have played a crucial role in shaping the paradigm of evaluating generalisation capabilities from multiple perspectives. Although the paradigm is shifting towards more fine-grained evaluation across diverse complex tasks, the delicate question of how to aggregate the performances has received particular interest in the community. The benchmarks generally follow the unspoken *utilitarian* principles, where the systems are ranked based on their mean average score over task-specific metrics. Such aggregation procedure has been viewed as a sub-optimal evaluation protocol, which may have created the illusion of progress in the field. This paper proposes **Vote'n'Rank**, a framework for ranking systems in multi-task benchmarks under the principles of the social choice theory. We demonstrate that our approach can be efficiently utilised to draw new insights on benchmarking in several AI sub-fields and identify the best-performing systems in simulated practical scenarios that meet user needs. The **Vote'n'Rank**'s procedures are empirically shown to be more robust than the mean average while being able to handle missing performance scores and determine conditions under which the system becomes the winner.

### Objective
Our main objective is to re-interpret the common benchmarking trends through the lens of the social choice theory and discuss how the leaderboards get changed if we follow the **Vote'n'Rank** rank aggregation principles. We also analyse the application of our methods to real-life scenarios, where the leaderboard holders do not report the performance for particular tasks, and the user can define the system evaluation criteria based on their end needs.

### Case studies
We consider four case studies (CSs) on three benchmarks across multiple ML fields: GLUE, SuperGLUE, and VALUE. The experiments can be reproduced using the following notebooks:

1. ```CS1 - Reranking Leaderboards.ipynb``` -- re-ranking the leaderboards and selecting the winner.
2. ```CS2 - Condorcet Search.ipynb``` -- identifying prosopective and non-prospective models.
3. ```CS3 - NaN Stability.ipynb``` -- exploring robustness to missing performance scores (replaced with NaNs).
4. ```CS4 - User-Oriented Setting.ipynb``` -- ranking systems in a simulated practical scenarios according to the following criteria: *performance*, *computational efficiency*, and *fairness*.

### Data and other materials
The publicly available leaderboard results (by the access date) and CS4 experiment results can be found [here](https://github.com/PragmaticsLab/vote_and_rank/tree/main/tables). 
