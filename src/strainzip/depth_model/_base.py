from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import jax.numpy as jnp
import numpy as np
from jax import hessian as jax_hessian
from jax.tree_util import Partial


@dataclass
class DepthModelResult:
    model: "DepthModel"
    params: Mapping[str, Any]
    X: Any
    y: Any
    converged: bool
    debug: Mapping[str, Any]

    @property
    def beta(self):
        return self.params["beta"]

    @property
    def residual(self):
        return self.y - self.X @ self.beta

    @property
    def stderr_beta(self):
        return self.model.stderr_beta(**self.params, y=self.y, X=self.X)

    @property
    def loglik(self):
        return self.model.loglik(**self.params, y=self.y, X=self.X)

    @property
    def num_paths(self):
        return self.X.shape[1]

    @property
    def num_params(self):
        return self.model.count_params(self.num_samples, self.num_edges, self.num_paths)

    @property
    def num_edges(self):
        return self.y.shape[0]

    @property
    def num_samples(self):
        return self.y.shape[1]

    def get_score(self, score_name):
        if score_name == "aic":
            return -self._aic
        elif score_name == "bic":
            return -self._bic
        elif score_name == "aicc":
            return -self._aicc
        else:
            raise ValueError(f"Requested {score_name} is not a known score name.")

    @property
    def _bic(self):
        num_observations = self.num_edges * self.num_samples
        bic = -2 * self.loglik + 2 * self.num_params * np.log(num_observations)
        return bic

    @property
    def _aic(self):
        aic = -2 * self.loglik + 2 * self.num_params
        return aic

    @property
    def _aicc(self):
        num_observations = self.num_edges * self.num_samples
        aicc = self._aic + (2 * self.num_params**2 + 2 * self.num_params) / (
            num_observations - self.num_params - 1
        )
        return aicc


class DepthModel:
    param_names = []

    def _fit(self, y, X) -> Tuple[Mapping[str, Any], bool, Any]:
        raise NotImplementedError(
            "Subclasses of DepthModel must implement the *_fit* method "
            "that returns a dictionary of estimated parameters. "
            "It also returns a second value (optional; may be None), which "
            "is passed on for debugging/information purposes. "
            "Implementations of DepthModel._fit are also responsible for "
            "tracking optimization error, checking for "
            "convergence, etc., and raising appropriate errors."
        )

    def loglik(self, beta, y, X, **params):
        raise NotImplementedError(
            "Subclasses of DepthModel must implement the *loglik* method that returns the parameter log-likelihood given the data."
        )

    def count_params(self, num_samples, num_edges, num_paths):
        raise NotImplementedError(
            "Subclasses of DepthModel must implement the *count_params* method that returns the number of parameters to use in score calculations."
        )

    def stderr_beta(self, beta, y, X, **params):
        raise NotImplementedError(
            "Subclasses of DepthModel must implement the *loglik* method that returns the parameter log-likelihood given the data."
        )

    def fit(self, y, X) -> DepthModelResult:
        params, converged, debug = self._fit(y, X)
        return DepthModelResult(
            model=self,
            params=params,
            X=X,
            y=y,
            converged=converged,
            debug=debug,
        )


class HessianDepthModel(DepthModel):
    def hessian_beta(self, beta, y, X, **params):
        raise NotImplementedError(
            "Subclasses of DepthModelWithHessian must implement the *hessian_beta* method that returns the hessian matrix across beta elements evaluated at a point."
        )

    def covariance_beta(self, beta, y, X, **params):
        try:
            cov = np.linalg.inv(self.hessian_beta(beta, y, X, **params))
        except np.linalg.LinAlgError:
            cov = np.nan * np.ones_like(self.hessian_beta(beta, y, X, **params))
        return cov

    def stderr_beta(self, beta, y, X, **params):
        stderr = np.sqrt(np.diag(self.covariance_beta(beta, y, X, **params))).reshape(
            beta.shape
        )
        return np.nan_to_num(stderr, nan=np.inf)


class JaxDepthModel(HessianDepthModel):
    def _jax_loglik(self, beta, y, X, **params):
        raise NotImplementedError(
            "Subclasses of JaxDepthModel must implement the *_jax_loglik* method that returns a JAX version of the loglik"
        )

    def loglik(self, *args, **kwargs):
        return self._jax_loglik(*args, **kwargs)

    def _negloglik(self, *args, **kwargs):
        return -self._jax_loglik(*args, **kwargs)

    def hessian_beta(self, beta, y, X, **params):
        num_betas = beta.shape[0] * beta.shape[1]
        hessian = jax_hessian(
            Partial(self._negloglik, y=y, X=X, **params), argnums=[0]
        )(beta)
        return hessian[0][0].reshape((num_betas, num_betas))
