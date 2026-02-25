#!/usr/bin/env python
"""Synthetic linear bandit experiment: meTS, meTSFactored, HierTS, LinUCB, LinTS."""

import argparse
import os
import sys

import numpy as np

# Package root (mixed_effect_ts)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandits import CoBandit, evaluate
from algorithms.linear import LinTS, LinUCB, HierTS, meTS, meTSFactored
from utils.plotting import plot_regret


def build_envs(K, L, d, sigma, num_runs):
    envs = []
    for _ in range(num_runs):
        contexts = 2 * np.random.rand(100, d) - 1
        A = 2 * np.random.rand(K, L) - 1
        matA = np.zeros((K * d, L * d))
        for i in range(K):
            matA[i * d : (i + 1) * d, :] = np.kron(A[i, :].T, np.eye(d))
        mu_psi = np.zeros(d * L)
        Sigma_psi = 3 * np.eye(d * L)
        Sigma0 = np.tile(np.eye(d), (K, 1, 1))
        mar_Theta0 = np.zeros((K, d))
        mar_Sigma0 = np.copy(Sigma0)
        for i in range(K):
            matAi = matA[i * d : (i + 1) * d, :]
            mar_Theta0[i, :] = matAi.dot(mu_psi)
            mar_Sigma0[i, :, :] = Sigma0[i, :, :] + matAi.dot(Sigma_psi).dot(matAi.T)
        Psi = np.random.multivariate_normal(mu_psi, Sigma_psi)
        matPsi = np.reshape(Psi, (L, d))
        Theta = np.random.randn(K, d)
        for i in range(K):
            Theta[i, :] = np.random.multivariate_normal(
                A[i, :].dot(matPsi), Sigma0[i, :, :]
            )
        env = CoBandit(K, contexts, Theta, sigma=sigma)
        env.A = A
        env.mu_psi = mu_psi
        env.Sigma_psi = Sigma_psi
        env.Sigma0 = Sigma0
        env.mar_Theta0 = mar_Theta0
        env.mar_Sigma0 = mar_Sigma0
        envs.append(env)
    return envs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--num_runs", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--save", type=str, default=None, help="Directory to save plot")
    args = ap.parse_args()

    algs = [
        (meTS, {}, "red", "-", "meTS-Lin"),
        (meTSFactored, {}, "orange", "-", "meTS-Lin-Fa"),
        (HierTS, {}, "gray", "-", "HierTS"),
        (LinUCB, {}, "cyan", "-", "LinUCB"),
        (LinTS, {}, "blue", "-", "LinTS"),
    ]

    envs = build_envs(args.K, args.L, args.d, args.sigma, args.num_runs)
    results = []
    for Alg, params, _c, _ls, label in algs:
        regret, _ = evaluate(Alg, params, envs, args.n)
        results.append(regret)

    title = "Linear Bandit, K = %d, L = %d, d = %d" % (args.K, args.L, args.d)
    save_path = None
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_path = os.path.join(
            args.save, "lin_K%d_L%d_d%d.png" % (args.K, args.L, args.d)
        )
    plot_regret(results, algs, title, save_path=save_path)
    if save_path:
        print("Saved plot to %s" % save_path)


if __name__ == "__main__":
    main()
