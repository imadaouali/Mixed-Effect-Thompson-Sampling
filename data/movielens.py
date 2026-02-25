"""MovieLens data loading and ALS matrix factorization."""

import os

import numpy as np


def load_ratings(path):
    """Load ratings from MovieLens ratings file; return M (dense matrix) and W (indicator)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Ratings file not found: %r. Download MovieLens 1M from "
            "https://grouplens.org/datasets/movielens/1m/, extract it, and run with "
            "--data /path/to/ml-1m/ratings.dat (or put ratings.dat in the current directory)."
            % path
        )
    # MovieLens 1M format: user::movie::rating::timestamp (np.loadtxt only allows single-char delimiter)
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 3:
                rows.append([int(parts[0]), int(parts[1]), int(parts[2])])
    D = np.array(rows, dtype=np.int64)
    num_users = D[:, 0].max()
    num_movies = D[:, 1].max()
    D[:, :2] = D[:, :2] - 1
    M = np.zeros((num_users, num_movies))
    M[D[:, 0], D[:, 1]] = D[:, 2]
    W = np.zeros((num_users, num_movies))
    W[D[:, 0], D[:, 1]] = 1
    ndx = np.random.permutation(num_users)
    M = M[ndx, :]
    W = W[ndx, :]
    return M, W


def ALS(M, W, d=10, num_iter=20, verbose=True):
    """Alternating Least Squares for matrix factorization. Returns (users, movies)."""
    num_rows = M.shape[0]
    num_cols = M.shape[1]
    U = 2 * np.random.rand(num_rows, d) - 1
    V = 2 * np.random.rand(num_cols, d) - 1
    reg = 1.0
    for it in range(num_iter):
        for i in range(num_rows):
            sel = np.flatnonzero(W[i, :])
            G = V[sel, :].T.dot(V[sel, :]) + reg * np.eye(d)
            U[i, :] = np.linalg.solve(G, V[sel, :].T.dot(M[i, sel]))
        for j in range(num_cols):
            sel = np.flatnonzero(W[:, j])
            G = U[sel, :].T.dot(U[sel, :]) + reg * np.eye(d)
            V[j, :] = np.linalg.solve(G, U[sel, :].T.dot(M[sel, j]))
        if verbose:
            err = np.linalg.norm(W * (M - U.dot(V.T))) / np.sqrt(W.sum())
            print("%.3f " % err, end="")
    if verbose:
        print()
    return U, V
