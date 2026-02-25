"""Evaluation: single and parallel runs of bandit algorithms."""

import time

import numpy as np
from joblib import Parallel, delayed


def evaluate_one(Alg, params, env, n, period_size=1):
    """One run of a bandit algorithm."""
    alg = Alg(env, n, params)
    regret = np.zeros(n // period_size)
    for t in range(n):
        env.randomize()
        arm = alg.get_arm(t)
        alg.update(t, arm, env.reward(arm))
        regret[t // period_size] += env.regret(arm)
    return regret, alg


def evaluate(Alg, params, env, n=1000, period_size=1, printout=True):
    """Multiple runs of a bandit algorithm (parallel over env list)."""
    if printout:
        print("Evaluating %s" % Alg.print(), end="")
    start = time.time()
    num_exps = len(env)
    regret = np.zeros((n // period_size, num_exps))
    alg = num_exps * [None]
    output = Parallel(n_jobs=-1)(
        delayed(evaluate_one)(Alg, params, env[ex], n, period_size)
        for ex in range(num_exps)
    )
    for ex in range(num_exps):
        regret[:, ex] = output[ex][0]
        alg[ex] = output[ex][1]
    if printout:
        print(" %.1f seconds" % (time.time() - start))
    if printout:
        total_regret = regret.sum(axis=0)
        print(
            "Regret: %.2f +/- %.2f (median: %.2f, max: %.2f, min: %.2f)"
            % (
                total_regret.mean(),
                total_regret.std() / np.sqrt(num_exps),
                np.median(total_regret),
                total_regret.max(),
                total_regret.min(),
            )
        )
    return regret, alg
