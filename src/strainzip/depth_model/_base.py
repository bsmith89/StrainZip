from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np


class BaseDepthModel:
    def fit(self, y, X):
        raise NotImplementedError(
            "Subclasses of DepthModel must implement the *fit* method."
        )


@dataclass
class FitResult:
    beta: Any
    sigma: Any
    hessian_func: Any
    loglik_func: Any
    X: Any
    y: Any
    alpha: float
    opt: Any

    @property
    def hessian_beta(self):
        num_betas = self.num_paths * self.num_samples
        return self.hessian_func(self.beta, self.sigma)[0][0].reshape(
            (num_betas, num_betas)
        )

    @property
    def loglik(self):
        return self.loglik_func(self.beta, self.sigma, self.y, self.X, self.alpha)

    @property
    def covariance_beta(self):
        try:
            cov = jnp.linalg.inv(self.hessian_beta)
        except np.linalg.LinAlgError:
            cov = np.nan * np.ones_like(self.hessian_beta)
        return cov

    @property
    def num_paths(self):
        return self.X.shape[1]

    @property
    def num_params(self):
        # # NOTE (2024-05-17): Experimenting with a global variance.
        # return self.num_paths * self.num_samples + 1
        return self.num_paths * self.num_samples + self.num_samples

    @property
    def num_edges(self):
        return self.y.shape[0]

    @property
    def num_samples(self):
        return self.y.shape[1]

    @property
    def score(self):
        return -self.bic

    def get_score(self, score_name):
        if score_name == "aic":
            return -self.aic
        elif score_name == "bic":
            return -self.bic
        elif score_name == "aicc":
            return -self.aicc
        else:
            raise ValueError(f"Requested {score_name} is not a known score name.")

    @property
    def bic(self):
        num_observations = self.num_edges * self.num_samples
        bic = -2 * self.loglik + 2 * self.num_params * jnp.log(num_observations)
        return bic

    @property
    def aic(self):
        aic = -2 * self.loglik + 2 * self.num_params
        return aic

    @property
    def aicc(self):
        num_observations = self.num_edges * self.num_samples
        aicc = self.aic + (2 * self.num_params**2 + 2 * self.num_params) / (
            num_observations - self.num_params - 1
        )
        return aicc

    @property
    def stderr_beta(self):
        return np.nan_to_num(
            jnp.sqrt(jnp.diag(self.covariance_beta)).reshape(self.beta.shape),
            nan=np.inf,
        )

    @property
    def residual(self):
        return self.y - self.X @ self.beta
