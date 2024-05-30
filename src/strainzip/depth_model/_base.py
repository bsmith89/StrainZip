from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np


@dataclass
class DepthModelResult:
    model: "BaseDepthModel"
    params: Mapping[str, Any]
    X: Any
    y: Any
    debug: Mapping[str, Any]

    @property
    def beta(self):
        return self.params["beta"]

    @property
    def hessian_beta(self):
        num_betas = self.num_paths * self.num_samples
        return self.model.hessian(
            beta=self.beta, y=self.y, X=self.X, sigma=self.params["sigma"]
        )[0][0].reshape((num_betas, num_betas))

    @property
    def loglik(self):
        return self.model.loglik(**self.params, y=self.y, X=self.X)

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
        bic = -2 * self.loglik + 2 * self.num_params * np.log(num_observations)
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


class BaseDepthModel:
    def fit(self, y, X) -> DepthModelResult:
        raise NotImplementedError(
            "Subclasses of DepthModel must implement the *fit* method that returns a model result."
        )

    def loglik(self, beta, y, X, **kwargs):
        raise NotImplementedError(
            "Subclasses of DepthModel must implement the *loglik* method that returns the parameter log-likelihood given the data."
        )

    def hessian(self, beta, y, X, **kwargs):
        raise NotImplementedError(
            "Subclasses of DepthModel must implement the *loglik* method that returns the parameter log-likelihood given the data."
        )
