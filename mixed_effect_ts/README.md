# [Mixed-Effect-Thompson-Sampling](https://arxiv.org/abs/2205.15124?context=cs)

Experiments for the paper [Mixed-Effect-Thompson-Sampling](https://arxiv.org/abs/2205.15124?context=cs)

**Imad AOUALI** (Criteo), Branislav Kveton (Amazon), Sumeet Katariya (Amazon)

## Abstract

A contextual bandit is a popular framework for online learning to act under uncertainty. In practice, the number of actions is huge and their expected rewards are correlated. In this work, we introduce a general framework for capturing such correlations through a mixed-effect model where actions are related through multiple shared effect parameters. To explore efficiently using this structure, we propose Mixed-Effect Thompson Sampling (meTS) and bound its Bayes regret. The regret bound has two terms, one for learning the action parameters and the other for learning the shared effect parameters. The terms reflect the structure of our model and the quality of priors. Our theoretical findings are validated empirically using both synthetic and real-world problems. We also propose numerous extensions of practical interest. While they do not come with guarantees, they perform well empirically and show the generality of the proposed framework.

## Setup

From the repository root (`mixed_effect_ts/`):

```bash
pip install -r requirements.txt
```

Requirements: `numpy`, `scipy`, `matplotlib`, `joblib`, `scikit-learn`. Optional: run `pip freeze > requirements.txt` in your virtualenv to pin exact versions.

**Optional:** The sigmoid used in logistic bandits is numerically stable by default (clipped). To match the original notebook formula exactly (may trigger overflow warnings), set `METS_LEGACY_SIGMOID=1`.

## Repository Structure

- **Notebooks** (original experiments):
  - `meTS-Lin.ipynb` — meTS on synthetic linear bandits
  - `meTS-Lin-MovieLens.ipynb` — meTS on MovieLens with linear rewards
  - `meTS-Log.ipynb` — meTS on synthetic logistic bandits
  - `meTS-Log-MovieLens.ipynb` — meTS on MovieLens with logistic rewards

- **Python package** (refactored):
  - `bandits/` — environments (`CoBandit`, `LogBandit`) and evaluation helpers
  - `algorithms/` — linear (`LinTS`, `LinUCB`, `meTS`, `meTSFactored`, `HierTS`) and logistic (`LogTS`, `LogUCB`, `UCBLog`, `meTS`, `FactoredmeTS`, `LinmeTS`, `HierTS`)
  - `data/` — MovieLens loading and ALS factorization
  - `utils/` — plotting utilities
  - `experiments/` — CLI scripts to run the same experiments as the notebooks

- **Data (MovieLens scripts):** Place `ratings.dat` in `data/` (default path), or download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and pass `--data /path/to/ml-1m/ratings.dat`.

## Usage

Run from the repository root (`mixed_effect_ts/`). Scripts add the package root to `sys.path` automatically.

**Synthetic linear bandit** (default: K=100, L=3, d=2, n=5000, 50 runs):

```bash
python experiments/run_lin.py --K 100 --L 3 --d 2 --n 5000 --num_runs 50 --sigma 1
python experiments/run_lin.py --save plots/
```

**MovieLens linear bandit** (uses `data/ratings.dat` by default):

```bash
python experiments/run_lin_movielens.py --K 100 --L 5 --d 2 --n 5000 --num_runs 50 --sigma 0.5
python experiments/run_lin_movielens.py --save plots/
```

**Synthetic logistic bandit**:

```bash
python experiments/run_log.py --K 100 --L 3 --d 2 --n 5000 --num_runs 50 --sigma 1
python experiments/run_log.py --save plots/
```

**MovieLens logistic bandit**:

```bash
python experiments/run_log_movielens.py --K 100 --L 5 --d 2 --n 5000 --num_runs 50
python experiments/run_log_movielens.py --save plots/
```

Use `--save <dir>` to write regret plots as PNGs; otherwise plots are shown interactively.

**Run all experiments overnight:** from `mixed_effect_ts/`, run `./run_all_overnight.sh`. This runs the four experiments in sequence, saves plots to `plots/`, and appends output to `run_all_overnight.log`. If one fails, the rest still run.

## License

MIT (see LICENSE).
