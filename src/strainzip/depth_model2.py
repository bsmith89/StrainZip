from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jaxopt
import numpy as np
from jax import hessian, jit
from jax.nn import softplus
from jax.scipy.stats.norm import logpdf as NormalLogPDF
from jax.tree_util import Partial

from strainzip.errors import ConvergenceException


@dataclass
class FitResult:
    beta: Any
    sigma: Any
    hessian_func: Any
    X: Any
    y: Any
    opt: Any

    @property
    def hessian_beta(self):
        num_betas = self.num_paths * self.num_samples
        return self.hessian_func(self.beta, self.sigma)[0][0].reshape(
            (num_betas, num_betas)
        )

    @property
    def loglik(self):
        return loglik(self.beta, self.sigma, self.y, self.X)

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


def loglik(beta, sigma, y, X):
    expect = X @ beta
    return NormalLogPDF(y, loc=expect, scale=sigma).sum()


def _negloglik(*args, **kwargs):
    return -1 * loglik(*args, **kwargs)


def _residual(beta, y, X):
    expect = X @ beta
    return y - expect


@jit
def _optimize(y, X, maxiter=500):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta_raw = jnp.zeros((p_paths, s_samples))

    def objective(beta_raw, y, X):
        beta = softplus(beta_raw)
        return (_residual(beta, y, X) ** 2).sum()

    # Estimate beta by minimizing the sum of squared residuals.
    est_beta_raw, opt = jaxopt.LBFGS(Partial(objective, y=y, X=X), maxiter=maxiter).run(
        init_params=init_beta_raw,
    )
    est_beta = softplus(est_beta_raw)

    # Estimate sigma as the root mean sum of squared residuals.
    est_sigma = jnp.sqrt((_residual(est_beta, y, X) ** 2).mean(0, keepdims=True))
    return (est_beta, est_sigma), opt


def fit(y, X, maxiter=500):
    params_est, opt = _optimize(y=y, X=X, maxiter=maxiter)
    if not opt.iter_num < maxiter:
        raise ConvergenceException(opt)

    # NOTE: Hessian of the *negative* log likelihood, because this is what's being
    # minimized? (How does this make sense??)
    hessian_func = hessian(
        Partial(_negloglik, y=y, X=X), argnums=range(len(params_est))
    )
    # FIXME: Doesn't generalize well to other models. I'll have to change this line.
    est_beta, est_sigma = params_est
    fit_result = FitResult(
        beta=est_beta,
        sigma=est_sigma,
        hessian_func=hessian_func,
        X=X,
        y=y,
        opt=opt,
    )
    return fit_result


class SoftPlusNormal:
    def __init__(self, maxiter=500):
        self.maxiter = maxiter

    def fit(self, y, X):
        return fit(y, X, self.maxiter)
