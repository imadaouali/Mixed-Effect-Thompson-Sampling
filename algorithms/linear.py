"""Linear bandit algorithms: LinTS, LinUCB, meTS, meTSFactored, HierTS."""

import numpy as np
from scipy.linalg import block_diag


class LinBanditAlg:
    def __init__(self, env, n, params):
        self.env = env
        self.K = self.env.K
        self.d = self.env.d
        self.n = n
        self.Theta0 = np.copy(env.mar_Theta0)
        self.Sigma0 = np.copy(env.mar_Sigma0)
        self.sigma = self.env.sigma
        for attr, val in params.items():
            if isinstance(val, np.ndarray):
                setattr(self, attr, np.copy(val))
            else:
                setattr(self, attr, val)
        self.Lambda0 = np.zeros((self.K, self.d, self.d))
        for i in range(self.K):
            self.Lambda0[i, :, :] = np.linalg.inv(self.Sigma0[i, :, :])
        self.G = np.zeros((self.K, self.d, self.d))
        self.B = np.zeros((self.K, self.d))

    def update(self, t, arm, r):
        x = self.env.X[arm, :]
        self.G[arm, :, :] += np.outer(x, x) / np.square(self.sigma)
        self.B[arm, :] += x * r / np.square(self.sigma)


class LinTS(LinBanditAlg):
    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        for i in range(self.K):
            Gt = self.Lambda0[i, :, :] + self.G[i, :, :]
            Sigma_hat = np.linalg.inv(Gt)
            theta_hat = np.linalg.solve(
                Gt,
                self.Lambda0[i, :, :].dot(self.Theta0[i, :]) + self.B[i, :],
            )
            theta_tilde = np.random.multivariate_normal(theta_hat, Sigma_hat)
            self.mu[i] = self.env.X[i, :].dot(theta_tilde)
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "LinTS"


class LinUCB(LinBanditAlg):
    def __init__(self, env, n, params):
        LinBanditAlg.__init__(self, env, n, params)
        self.cew = self.confidence_ellipsoid_width(n)

    def confidence_ellipsoid_width(self, t):
        delta = 1 / self.n
        L = np.amax(np.linalg.norm(self.env.contexts, axis=1))
        Lambda = np.trace(self.Lambda0, axis1=-2, axis2=-1).max() / self.d
        R = self.sigma
        S = np.sqrt(self.d)
        width = np.sqrt(Lambda) * S + R * np.sqrt(
            self.d * np.log((1 + t * np.square(L) / Lambda) / delta)
        )
        return width

    def get_arm(self, t):
        self.mu = np.zeros(self.K)
        for i in range(self.K):
            Gt = self.Lambda0[i, :, :] + self.G[i, :, :]
            Sigma_hat = np.linalg.inv(Gt)
            theta_hat = np.linalg.solve(
                Gt,
                self.Lambda0[i, :, :].dot(self.Theta0[i, :]) + self.B[i, :],
            )
            Sigma_hat /= np.square(self.sigma)
            self.mu[i] = self.env.X[i, :].dot(theta_hat) + self.cew * np.sqrt(
                self.env.X[i, :].dot(Sigma_hat).dot(self.env.X[i, :])
            )
        return np.argmax(self.mu)

    @staticmethod
    def print():
        return "LinUCB"


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
        return "meTS"


class meTSFactored:
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
        return "meTSFactored"


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
