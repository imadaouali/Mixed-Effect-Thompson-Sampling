# Mixed-Effect-Thompson-Sampling

[Imad AOUALI](https://www.linkedin.com/in/imad-aouali/) (Criteo), Branislav Kveton (Amazon), Sumeet Katariya (Amazon)

## Abstract

A contextual bandit is a popular framework for online learning to act under uncertainty. In practice, the number of actions is huge and their expected rewards are correlated. In this work, we introduce a general framework for capturing such correlations through a mixed-effect model where actions are related through multiple shared effect parameters. To explore efficiently using this structure, we propose Mixed-Effect Thompson Sampling (meTS) and bound its Bayes regret. The regret bound has two terms, one for learning the action parameters and the other for learning the shared effect parameters. The terms reflect the structure of our model and the quality of priors. Our theoretical findings are validated empirically using both synthetic and real-world problems. We also propose numerous extensions of practical interest. While they do not come with guarantees, they perform well empirically and show the generality of the proposed framework.

## Repository Structure

This repository is structured as follows

- `meTS-Lin.ipynb`
meTS experiments on synthetic linear bandit problems

- `meTS-Lin-MovieLens.ipynb`
meTS experiments on MovieLens dataset with linear rewards

- `meTS-Log.ipynb` 
meTS experiments on synthetic logistic bandit problems

- `meTS-Log-MovieLens.ipynb`
meTS experiments on MovieLens dataset with logistic rewards

`ratings.dat`
MovieLens 1M dataset from https://grouplens.org/datasets/movielens/1m/ 

[imad-email]: mailto:imadaouali9@gmail.com 
