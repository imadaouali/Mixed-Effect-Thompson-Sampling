"""Bandit environments: linear (CoBandit) and logistic (LogBandit)."""

import os

import numpy as np

_USE_STABLE_SIGMOID = os.environ.get("METS_LEGACY_SIGMOID", "").strip() not in ("1", "true", "yes")


class CoBandit:
    """Contextual bandit with K arms and linear rewards."""

    def __init__(self, K, contexts, Theta, sigma=1.0):
        self.K = K
        self.contexts = np.copy(contexts)
        self.num_contexts = self.contexts.shape[0]
        self.d = self.contexts.shape[1]
        self.Theta = np.copy(Theta)
        self.sigma = sigma
        self.randomize()

    def randomize(self):
        ndx = np.random.randint(self.num_contexts, size=self.K)
        self.X = self.contexts[ndx, :]
        self.mut = (self.X * self.Theta).sum(axis=-1)
        self.rt = self.mut + self.sigma * np.random.randn(self.K)
        self.best_arm = np.argmax(self.mut)

    def reward(self, arm):
        return self.rt[arm]

    def regret(self, arm):
        return self.rt[self.best_arm] - self.rt[arm]

    def pregret(self, arm):
        return self.mut[self.best_arm] - self.mut[arm]

    def print(self):
        return "Contextual bandit: %d dimensions, %d arms" % (self.d, self.K)


class LogBandit:
    """Logistic bandit with K arms and binary rewards."""

    def __init__(self, K, contexts, Theta, sigma):
        self.K = K
        self.contexts = np.copy(contexts)
        self.num_contexts = self.contexts.shape[0]
        self.d = self.contexts.shape[1]
        self.Theta = np.copy(Theta)
        self.sigma = sigma
        self.randomize()

    def sigmoid(self, x):
        if _USE_STABLE_SIGMOID:
            x = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-x))

    def randomize(self):
        self.ndx = np.random.randint(self.num_contexts, size=self.K)
        self.X = self.contexts[self.ndx, :]
        self.mut = self.sigmoid((self.X * self.Theta).sum(axis=-1))
        self.rt = (np.random.rand(self.K) < self.mut).astype(float)
        self.best_arm = np.argmax(self.mut)

    def reward(self, arm):
        return self.rt[arm]

    def regret(self, arm):
        return self.rt[self.best_arm] - self.rt[arm]

    def pregret(self, arm):
        return self.mut[self.best_arm] - self.mut[arm]

    def print(self):
        return "Logistic bandit: %d dimensions, %d arms" % (self.d, self.K)
