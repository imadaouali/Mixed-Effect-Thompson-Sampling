"""Logistic bandit algorithms: LogTS, LogUCB, UCBLog, meTS, FactoredmeTS, LinmeTS, HierTS."""

import os

import numpy as np
from scipy.linalg import block_diag

# Use stable sigmoid by default. Set METS_LEGACY_SIGMOID=1 for unclipped notebook formula (may trigger overflow warnings).
_USE_STABLE_SIGMOID = os.environ.get("METS_LEGACY_SIGMOID", "").strip() not in ("1", "true", "yes")


def _sigmoid_stable(x):
    """Numerically stable sigmoid; use when METS_STABLE_SIGMOID=1."""
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid(x):
    """Sigmoid: stable if METS_STABLE_SIGMOID is set, else notebook formula."""
    if _USE_STABLE_SIGMOID:
        return _sigmoid_stable(x)
    return 1 / (1 + np.exp(-x))


class LogBanditAlg:
    def __init__(self, env, n, params):
        self.env = env
        self.K = self.env.K
        self.d = self.env.d
        self.n = n
        self.Theta0 = np.copy(env.mar_Theta0)
        self.Sigma0 = np.copy(env.mar_Sigma0)
        self.sigma = self.env.sigma
        self.irls_Theta = np.copy(env.mar_Theta0)
        self.irls_error = 1e-3
        self.irls_num_iter = 1000
        self.batch_size = 30
        for attr, val in params.items():
            setattr(self, attr, val)
        self.pulls = np.zeros(self.K, dtype=int)
        self.Lambda0 = np.zeros((self.K, self.d, self.d))
        for i in range(self.K):
            self.Lambda0[i, :, :] = np.linalg.inv(self.Sigma0[i, :, :])
        self.G = np.zeros((self.K, self.d, self.d))
        self.context_ndx = [[] for _ in range(self.K)]
        self.r = [[] for _ in range(self.K)]

    def update(self, t, arm, r):
        self.r[arm].append(r)
        self.pulls[arm] += 1
        self.context_ndx[arm].append(self.env.ndx[arm])
        x = self.env.X[arm, :]
        self.G[arm, :, :] += np.outer(x, x) / np.square(self.sigma)

    def sigmoid(self, x):
        return _sigmoid(x)

    def solve(self, arm):
        small_eye = 1e-5 * np.eye(self.d)
        theta = np.copy(self.irls_Theta[arm, :])
        nbr_obs = self.pulls[arm]
        t = self.r[arm]
        context_ndx = self.context_ndx[arm]
        num_iter = 0
        while num_iter < self.irls_num_iter:
            theta_old = np.copy(theta)
            Gram = small_eye.copy()
            Phiyt = np.zeros(self.d)
            if nbr_obs <= self.batch_size:
                batch = np.arange(nbr_obs)
            else:
                batch = np.random.choice(nbr_obs, size=self.batch_size)
            for i in batch:
                x = self.env.contexts[context_ndx[i], :]
                y = self.sigmoid((x * theta).sum())
                Gram += y * (1 - y) * np.outer(x, x)
                Phiyt += (y - t[i]) * x
            PhiRz = Gram.dot(theta) - Phiyt
            theta = np.linalg.solve(Gram, PhiRz)
            if np.linalg.norm(theta - theta_old) < self.irls_error:
                break
            num_iter += 1
        if num_iter == self.irls_num_iter:
            self.irls_Theta[arm, :] = self.Theta0[arm, :]
        else:
            self.irls_Theta[arm, :] = np.copy(theta)
        return theta, Gram


class LogUCB(LogBanditAlg):
    def __init__(self, env, n, params):
        LogBanditAlg.__init__(self, env, n, params)
        self.cew = self.confidence_ellipsoid_width(n)
        self.inv_Gt = np.zeros((self.K, self.d, self.d))

    def confidence_ellipsoid_width(self, t):
        delta = 1 / self.n
        c_m = np.amax(np.linalg.norm(self.env.contexts, axis=1))
        c_mu = 0.25
        k_mu = 0.25
        kappa = np.sqrt(3 + 2 * np.log(1 + 2 * np.square(c_m / self.sigma)))
        R_max = 1.0
        width = (2 * k_mu * kappa * R_max / c_mu) * np.sqrt(
            2 * self.d * np.log(t) * np.log(2 * self.d * self.n / delta)
        )
        return width

    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        if t == 0:
            for i in range(self.K):
                Gt = self.Lambda0[i, :, :] + self.G[i, :, :]
                self.inv_Gt[i, :, :] = np.linalg.inv(Gt)
                self.solve(i)
        else:
            Gt = self.Lambda0[self.At, :, :] + self.G[self.At, :, :]
            self.inv_Gt[self.At, :, :] = np.linalg.inv(Gt)
            self.solve(self.At)
        for i in range(self.K):
            theta_hat = self.irls_Theta[i, :]
            inv_Gt = self.inv_Gt[i, :, :]
            self.mu[i] = self.sigmoid(
                self.env.X[i, :].dot(theta_hat)
            ) + self.cew * np.sqrt(
                self.env.X[i, :].dot(inv_Gt).dot(self.env.X[i, :])
            )
        arm = np.argmax(self.mu)
        self.At = arm
        return arm

    @staticmethod
    def print():
        return "GLM-UCB"


class UCBLog(LogBanditAlg):
    def __init__(self, env, n, params):
        LogBanditAlg.__init__(self, env, n, params)
        self.cew = self.confidence_ellipsoid_width(n)
        self.inv_Gt = np.zeros((self.K, self.d, self.d))

    def confidence_ellipsoid_width(self, t):
        delta = 1 / self.n
        sigma = 0.5
        kappa = 0.25
        width = (sigma / kappa) * np.sqrt(
            (self.d / 2) * np.log(1 + 2 * self.n / self.d) + np.log(1 / delta)
        )
        return width

    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        if t == 0:
            for i in range(self.K):
                Gt = self.Lambda0[i, :, :] + self.G[i, :, :]
                self.inv_Gt[i, :, :] = np.linalg.inv(Gt)
                self.solve(i)
        else:
            Gt = self.Lambda0[self.At, :, :] + self.G[self.At, :, :]
            self.inv_Gt[self.At, :, :] = np.linalg.inv(Gt)
            self.solve(self.At)
        for i in range(self.K):
            theta_hat = self.irls_Theta[i, :]
            inv_Gt = self.inv_Gt[i, :, :]
            self.mu[i] = self.sigmoid(
                self.env.X[i, :].dot(theta_hat)
            ) + self.cew * np.sqrt(
                self.env.X[i, :].dot(inv_Gt).dot(self.env.X[i, :])
            )
        arm = np.argmax(self.mu)
        self.At = arm
        return arm

    @staticmethod
    def print():
        return "UCB-GLM"


class LogTS(LogBanditAlg):
    def __init__(self, env, n, params):
        LogBanditAlg.__init__(self, env, n, params)
        self.Grams = np.zeros((self.K, self.d, self.d))

    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        if t == 0:
            for i in range(self.K):
                _, Gram = self.solve(i)
                self.Grams[i, :, :] = Gram
        else:
            _, Gram = self.solve(self.At)
            self.Grams[self.At, :, :] = Gram
        for i in range(self.K):
            Sigma_ti = np.linalg.inv(
                self.Lambda0[i, :, :] + self.Grams[i, :, :]
            )
            mu_ti = self.Lambda0[i, :, :].dot(self.Theta0[i, :]) + self.Grams[
                i, :, :
            ].dot(self.irls_Theta[i, :])
            mu_ti = Sigma_ti.dot(mu_ti)
            theta_tilde = np.random.multivariate_normal(mu_ti, Sigma_ti)
            self.mu[i] = self.env.X[i, :].dot(theta_tilde)
        arm = np.argmax(self.mu)
        self.At = arm
        return arm

    @staticmethod
    def print():
        return "GLM-TSL"


class meTS:
    def __init__(self, env, n, params):
        self.env = env
        self.K = self.env.K
        self.d = self.env.d
        self.n = n
        self.A = self.env.A
        self.L = self.A.shape[1]
        self.mu_psi = np.copy(self.env.mu_psi)
        self.Sigma_psi = np.copy(self.env.Sigma_psi)
        self.Sigma0 = np.copy(self.env.Sigma0)
        self.sigma = self.env.sigma
        self.irls_Theta = np.zeros((self.K, self.d))
        self.irls_error = 1e-3
        self.irls_num_iter = 1000
        self.batch_size = 30
        for attr, val in params.items():
            if isinstance(val, np.ndarray):
                setattr(self, attr, np.copy(val))
            else:
                setattr(self, attr, val)
        self.Lambda_psi = np.linalg.inv(self.Sigma_psi)
        self.Lambda0 = np.zeros((self.K, self.d, self.d))
        for i in range(self.K):
            self.Lambda0[i, :, :] = np.linalg.inv(self.Sigma0[i, :, :])
        self.Theta0 = np.copy(self.env.mar_Theta0)
        self.G = np.zeros((self.K, self.d, self.d))
        self.B = np.zeros((self.K, self.d))
        self.pulls = np.zeros(self.K, dtype=int)
        self.context_ndx = [[] for _ in range(self.K)]
        self.r = [[] for _ in range(self.K)]

    def sigmoid(self, x):
        return _sigmoid(x)

    def solve(self, arm):
        small_eye = 1e-5 * np.eye(self.d)
        theta = np.copy(self.irls_Theta[arm, :])
        nbr_obs = self.pulls[arm]
        t = self.r[arm]
        context_ndx = self.context_ndx[arm]
        num_iter = 0
        while num_iter < self.irls_num_iter:
            theta_old = np.copy(theta)
            Gram = small_eye.copy()
            Phiyt = np.zeros(self.d)
            if nbr_obs <= self.batch_size:
                batch = np.arange(nbr_obs)
            else:
                batch = np.random.choice(nbr_obs, size=self.batch_size)
            for i in batch:
                x = self.env.contexts[context_ndx[i], :]
                y = self.sigmoid((x * theta).sum())
                Gram += y * (1 - y) * np.outer(x, x)
                Phiyt += (y - t[i]) * x
            PhiRz = Gram.dot(theta) - Phiyt
            theta = np.linalg.solve(Gram, PhiRz)
            if np.linalg.norm(theta - theta_old) < self.irls_error:
                break
            num_iter += 1
        if num_iter == self.irls_num_iter:
            self.irls_Theta[arm, :] = self.Theta0[arm, :]
        else:
            self.irls_Theta[arm, :] = np.copy(theta)
        return theta, Gram

    def update(self, t, arm, r):
        self.r[arm].append(r)
        self.pulls[arm] += 1
        self.context_ndx[arm].append(self.env.ndx[arm])
        theta, self.G[arm, :, :] = self.solve(arm)
        self.B[arm, :] = self.G[arm, :, :].dot(theta)

    def get_arm(self, t):
        small_eye = 1e-5 * np.eye(self.d)
        Lambda_t = np.copy(self.Lambda_psi)
        mu_t = self.Lambda_psi.dot(self.mu_psi)
        for i in range(self.K):
            aiai = np.outer(self.A[i, :], self.A[i, :])
            inv_Gi = np.linalg.inv(self.G[i, :, :] + small_eye)
            prior_adjusted_Gi = np.linalg.inv(self.Sigma0[i, :, :] + inv_Gi)
            Lambda_t += np.kron(aiai, prior_adjusted_Gi)
            prior_adjusted_Bi = prior_adjusted_Gi.dot(inv_Gi.dot(self.B[i, :]))
            mu_t += np.outer(self.A[i, :], prior_adjusted_Bi).flatten()
        Sigma_t = np.linalg.inv(Lambda_t)
        mu_t = Sigma_t.dot(mu_t)
        Psi_tilde = np.random.multivariate_normal(mu_t, Sigma_t)
        matPsi_tilde = np.reshape(Psi_tilde, (self.L, self.d))
        self.mu = np.zeros(self.K)
        for i in range(self.K):
            Sigma_ti = np.linalg.inv(self.Lambda0[i, :, :] + self.G[i, :, :])
            self.Theta0[i, :] = self.A[i, :].dot(matPsi_tilde)
            mu_ti = self.Lambda0[i, :, :].dot(self.Theta0[i, :]) + self.B[i, :]
            mu_ti = Sigma_ti.dot(mu_ti)
            theta_tilde = np.random.multivariate_normal(mu_ti, Sigma_ti)
            self.mu[i] = self.env.X[i, :].dot(theta_tilde)
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "meTS-GLM"


class FactoredmeTS:
    def __init__(self, env, n, params):
        self.env = env
        self.K = self.env.K
        self.d = self.env.d
        self.n = n
        self.A = self.env.A
        self.L = self.A.shape[1]
        self.mu_psi = np.copy(self.env.mu_psi)
        self.Sigma_psi = np.copy(self.env.Sigma_psi)
        self.Sigma0 = np.copy(self.env.Sigma0)
        self.sigma = self.env.sigma
        self.irls_Theta = np.zeros((self.K, self.d))
        self.irls_error = 1e-3
        self.irls_num_iter = 1000
        self.batch_size = 30
        for attr, val in params.items():
            if isinstance(val, np.ndarray):
                setattr(self, attr, np.copy(val))
            else:
                setattr(self, attr, val)
        self.Lambda_psi = np.linalg.inv(self.Sigma_psi)
        self.Lambda0 = np.zeros((self.K, self.d, self.d))
        for i in range(self.K):
            self.Lambda0[i, :, :] = np.linalg.inv(self.Sigma0[i, :, :])
        self.Theta0 = np.copy(self.env.mar_Theta0)
        self.G = np.zeros((self.K, self.d, self.d))
        self.B = np.zeros((self.K, self.d))
        self.pulls = np.zeros(self.K, dtype=int)
        self.context_ndx = [[] for _ in range(self.K)]
        self.r = [[] for _ in range(self.K)]

    def sigmoid(self, x):
        return _sigmoid(x)

    def solve(self, arm):
        small_eye = 1e-5 * np.eye(self.d)
        theta = np.copy(self.irls_Theta[arm, :])
        nbr_obs = self.pulls[arm]
        t = self.r[arm]
        context_ndx = self.context_ndx[arm]
        num_iter = 0
        while num_iter < self.irls_num_iter:
            theta_old = np.copy(theta)
            Gram = small_eye.copy()
            Phiyt = np.zeros(self.d)
            if nbr_obs <= self.batch_size:
                batch = np.arange(nbr_obs)
            else:
                batch = np.random.choice(nbr_obs, size=self.batch_size)
            for i in batch:
                x = self.env.contexts[context_ndx[i], :]
                y = self.sigmoid((x * theta).sum())
                Gram += y * (1 - y) * np.outer(x, x)
                Phiyt += (y - t[i]) * x
            PhiRz = Gram.dot(theta) - Phiyt
            theta = np.linalg.solve(Gram, PhiRz)
            if np.linalg.norm(theta - theta_old) < self.irls_error:
                break
            num_iter += 1
        if num_iter == self.irls_num_iter:
            self.irls_Theta[arm, :] = self.Theta0[arm, :]
        else:
            self.irls_Theta[arm, :] = np.copy(theta)
        return theta, Gram

    def update(self, t, arm, r):
        self.r[arm].append(r)
        self.pulls[arm] += 1
        self.context_ndx[arm].append(self.env.ndx[arm])
        theta, self.G[arm, :, :] = self.solve(arm)
        self.B[arm, :] = self.G[arm, :, :].dot(theta)

    def get_arm(self, t):
        small_eye = 1e-5 * np.eye(self.d)
        Lambda_t = []
        mu_t = []
        for l in range(self.L):
            Lambda_psi_l = self.Lambda_psi[
                l * self.d : (l + 1) * self.d, l * self.d : (l + 1) * self.d
            ]
            Lambda_t.append(np.copy(Lambda_psi_l))
            mu_t.append(Lambda_psi_l.dot(self.mu_psi[l * self.d : (l + 1) * self.d]))
        for i in range(self.K):
            inv_Gi = np.linalg.inv(self.G[i, :, :] + small_eye)
            prior_adjusted_Gi = np.linalg.inv(self.Sigma0[i, :, :] + inv_Gi)
            prior_adjusted_Bi = prior_adjusted_Gi.dot(inv_Gi.dot(self.B[i, :]))
            for l in range(self.L):
                Lambda_t[l] += (self.A[i, l] ** 2) * prior_adjusted_Gi
                mu_t[l] += self.A[i, l] * prior_adjusted_Bi
        Sigma_t = []
        for l in range(self.L):
            Sigma_t.append(np.linalg.inv(Lambda_t[l]))
            mu_t[l] = Sigma_t[l].dot(mu_t[l])
        Sigma_t = block_diag(*Sigma_t)
        mu_t = np.concatenate(mu_t)
        Psi_tilde = np.random.multivariate_normal(mu_t, Sigma_t)
        matPsi_tilde = np.reshape(Psi_tilde, (self.L, self.d))
        self.mu = np.zeros(self.K)
        for i in range(self.K):
            Sigma_ti = np.linalg.inv(self.Lambda0[i, :, :] + self.G[i, :, :])
            self.Theta0[i, :] = self.A[i, :].dot(matPsi_tilde)
            mu_ti = self.Lambda0[i, :, :].dot(self.Theta0[i, :]) + self.B[i, :]
            mu_ti = Sigma_ti.dot(mu_ti)
            theta_tilde = np.random.multivariate_normal(mu_ti, Sigma_ti)
            self.mu[i] = self.env.X[i, :].dot(theta_tilde)
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "meTS-GLM-Fa"


class LinmeTS:
    def __init__(self, env, n, params):
        self.env = env
        self.K = self.env.K
        self.d = self.env.d
        self.n = n
        self.A = self.env.A
        self.L = self.A.shape[1]
        self.mu_psi = np.copy(self.env.mu_psi)
        self.Sigma_psi = np.copy(self.env.Sigma_psi)
        self.Sigma0 = np.copy(self.env.Sigma0)
        self.sigma = self.env.sigma
        for attr, val in params.items():
            if isinstance(val, np.ndarray):
                setattr(self, attr, np.copy(val))
            else:
                setattr(self, attr, val)
        self.Lambda_psi = np.linalg.inv(self.Sigma_psi)
        self.Lambda0 = np.zeros((self.K, self.d, self.d))
        for i in range(self.K):
            self.Lambda0[i, :, :] = np.linalg.inv(self.Sigma0[i, :, :])
        self.G = np.zeros((self.K, self.d, self.d))
        self.B = np.zeros((self.K, self.d))

    def update(self, t, arm, r):
        x = self.env.X[arm, :]
        self.G[arm, :, :] += np.outer(x, x) / np.square(self.sigma)
        self.B[arm, :] += x * r / np.square(self.sigma)

    def get_arm(self, t):
        small_eye = 1e-3 * np.eye(self.d)
        Lambda_t = np.copy(self.Lambda_psi)
        mu_t = self.Lambda_psi.dot(self.mu_psi)
        for i in range(self.K):
            aiai = np.outer(self.A[i, :], self.A[i, :])
            inv_Gi = np.linalg.inv(self.G[i, :, :] + small_eye)
            prior_adjusted_Gi = np.linalg.inv(self.Sigma0[i, :, :] + inv_Gi)
            Lambda_t += np.kron(aiai, prior_adjusted_Gi)
            prior_adjusted_Bi = prior_adjusted_Gi.dot(inv_Gi.dot(self.B[i, :]))
            mu_t += np.outer(self.A[i, :], prior_adjusted_Bi).flatten()
        Sigma_t = np.linalg.inv(Lambda_t)
        mu_t = Sigma_t.dot(mu_t)
        Psi_tilde = np.random.multivariate_normal(mu_t, Sigma_t)
        matPsi_tilde = np.reshape(Psi_tilde, (self.L, self.d))
        self.mu = np.zeros(self.K)
        for i in range(self.K):
            Sigma_ti = np.linalg.inv(self.Lambda0[i, :, :] + self.G[i, :, :])
            mu_ti = (
                self.Lambda0[i, :, :].dot(self.A[i, :].dot(matPsi_tilde))
                + self.B[i, :]
            )
            mu_ti = Sigma_ti.dot(mu_ti)
            theta_tilde = np.random.multivariate_normal(mu_ti, Sigma_ti)
            self.mu[i] = self.env.X[i, :].dot(theta_tilde)
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "meTS-Lin"


class HierTS:
    def __init__(self, env, n, params):
        self.env = env
        self.K = self.env.K
        self.d = self.env.d
        self.n = n
        self.A = self.env.A
        self.L = self.A.shape[1]
        self.mu_psi = np.mean(
            self.env.mu_psi.reshape(self.L, self.d), axis=0
        )
        self.Sigma_psi = (1 / (self.L ** 2)) * np.sum(
            [
                self.env.Sigma_psi[
                    l * self.d : (l + 1) * self.d,
                    l * self.d : (l + 1) * self.d,
                ]
                for l in range(self.L)
            ],
            axis=0,
        )
        self.Sigma0 = np.copy(self.env.Sigma0)
        self.sigma = self.env.sigma
        for attr, val in params.items():
            if isinstance(val, np.ndarray):
                setattr(self, attr, np.copy(val))
            else:
                setattr(self, attr, val)
        self.Lambda_psi = np.linalg.inv(self.Sigma_psi)
        self.Lambda0 = np.zeros((self.K, self.d, self.d))
        for i in range(self.K):
            self.Lambda0[i, :, :] = np.linalg.inv(self.Sigma0[i, :, :])
        self.G = np.zeros((self.K, self.d, self.d))
        self.B = np.zeros((self.K, self.d))

    def update(self, t, arm, r):
        x = self.env.X[arm, :]
        self.G[arm, :, :] += np.outer(x, x) / np.square(self.sigma)
        self.B[arm, :] += x * r / np.square(self.sigma)

    def get_arm(self, t):
        small_eye = 1e-3 * np.eye(self.d)
        Lambda_t = np.copy(self.Lambda_psi)
        mu_t = self.Lambda_psi.dot(self.mu_psi)
        for i in range(self.K):
            inv_Gi = np.linalg.inv(self.G[i, :, :] + small_eye)
            prior_adjusted_Gi = np.linalg.inv(self.Sigma0[i, :, :] + inv_Gi)
            Lambda_t += prior_adjusted_Gi
            prior_adjusted_Bi = prior_adjusted_Gi.dot(inv_Gi.dot(self.B[i, :]))
            mu_t += prior_adjusted_Bi
        Sigma_t = np.linalg.inv(Lambda_t)
        mu_t = Sigma_t.dot(mu_t)
        Psi_tilde = np.random.multivariate_normal(mu_t, Sigma_t)
        self.mu = np.zeros(self.K)
        for i in range(self.K):
            Sigma_ti = np.linalg.inv(self.Lambda0[i, :, :] + self.G[i, :, :])
            mu_ti = self.Lambda0[i, :, :].dot(Psi_tilde) + self.B[i, :]
            mu_ti = Sigma_ti.dot(mu_ti)
            theta_tilde = np.random.multivariate_normal(mu_ti, Sigma_ti)
            self.mu[i] = self.env.X[i, :].dot(theta_tilde)
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "HierTS"
