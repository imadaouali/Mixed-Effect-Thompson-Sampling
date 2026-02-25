#!/usr/bin/env python
"""MovieLens logistic bandit experiment."""

import argparse
import os
import sys

import numpy as np
from sklearn.mixture import GaussianMixture

_package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _package_root)

from bandits import LogBandit, evaluate
from algorithms.logistic import meTS, FactoredmeTS, LinmeTS, HierTS, LogTS
from data.movielens import ALS, load_ratings
from utils.plotting import plot_regret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        default=os.path.join(_package_root, "data", "ratings.dat"),
    )
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--als_d", type=int, default=5)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--num_runs", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--verbose_als", action="store_true")
    args = ap.parse_args()

    M, W = load_ratings(args.data)
    users, movies = ALS(M, W, d=args.als_d, verbose=args.verbose_als)
    np.random.shuffle(users)
    np.random.shuffle(movies)
    users = users[:, : args.d]
    movies = movies[:, : args.d]
    mu_movie = movies.mean(axis=0)
    var_movie = movies.var(axis=0)

    algs = [
        (meTS, {}, "red", "-", "meTS-GLM"),
        (FactoredmeTS, {}, "orange", "-", "meTS-GLM-Fa"),
        (LinmeTS, {}, "purple", "-", "meTS-Lin"),
        (HierTS, {}, "gray", "-", "HierTS"),
        (LogTS, {}, "blue", "-", "GLM-TS"),
    ]

    # MovieLens logistic uses batch_size=100 in algorithms
    envs = []
    for _ in range(args.num_runs):
        contexts = users[np.random.randint(users.shape[0], size=100), :]
        sub = np.random.randint(movies.shape[0], size=args.K)
        Theta = movies[sub, :]
        gm = GaussianMixture(n_components=args.L, covariance_type="diag").fit(movies)
        matPsi = gm.means_
        mu_psi = matPsi.flatten()
        A = gm.predict_proba(Theta)
        Sigma_psi = 0.75 * np.diag(np.tile(var_movie, args.L))
        Sigma0 = 0.25 * np.tile(np.diag(var_movie), (args.K, 1, 1))
        mar_Theta0 = np.outer(np.ones(args.K), mu_movie)
        mar_Sigma0 = np.tile(np.diag(var_movie), (args.K, 1, 1))
        env = LogBandit(args.K, contexts, Theta, sigma=args.sigma)
        env.A = A
        env.mu_psi = mu_psi
        env.Sigma_psi = Sigma_psi
        env.Sigma0 = Sigma0
        env.mar_Theta0 = mar_Theta0
        env.mar_Sigma0 = mar_Sigma0
        envs.append(env)

    # Override batch_size for MovieLens (100 in notebook)
    results = []
    for Alg, params, _c, _ls, label in algs:
        p = dict(params)
        p["batch_size"] = 100
        regret, _ = evaluate(Alg, p, envs, args.n)
        results.append(regret)

    title = "K = %d, L = %d, d = %d" % (args.K, args.L, args.d)
    save_path = None
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_path = os.path.join(
            args.save, "log_movielens_K%d_L%d_d%d.png" % (args.K, args.L, args.d)
        )
    plot_regret(results, algs, title, save_path=save_path)
    if save_path:
        print("Saved plot to %s" % save_path)


if __name__ == "__main__":
    main()
