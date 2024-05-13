from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jaxopt
from jax import hessian, jit
from jax.nn import softplus
from jax.scipy.stats.norm import logpdf as NormalLogPDF
from jax.tree_util import Partial

from strainzip.errors import ConvergenceException


@dataclass
class FitResult:
    beta: Any
    sigma: Any
    score: float
    hessian_func: Any
    X: Any
    y: Any
    opt: Any

    @property
    def hessian_beta(self):
        p_paths, s_samples = self.beta.shape
        k_params = p_paths * s_samples
        return self.hessian_func(self.beta, self.sigma)[0][0].reshape(
            (k_params, k_params)
        )

    @property
    def covariance_beta(self):
        cov = jnp.linalg.inv(self.hessian_beta)
        return cov

    @property
    def stderr_beta(self):
        return jnp.sqrt(jnp.diag(self.covariance_beta)).reshape(self.beta.shape)

    @property
    def residual(self):
        return self.y - self.X @ self.beta


def loglik(beta, sigma, y, X):
    expect = X @ beta
    return NormalLogPDF(y, loc=expect, scale=sigma).sum()


def _negloglik(*args, **kwargs):
    return -1 * loglik(*args, **kwargs)


def _score(beta, sigma, y, X):
    e_edges, s_samples = y.shape
    _, p_paths = X.shape

    ll = loglik(beta, sigma, y, X)
    n = e_edges * s_samples
    k = p_paths * s_samples + s_samples
    # NOTE: This parameter count is only correct if fitting both beta and sigma.
    # However, it shouldn't matter for inter-comparison of this same depth model.
    # bic = -2 * ll + 2 * k * jnp.log(n)  # TODO: Consier using BIC instead?
    # aic = -2 * ll + 2 * k
    bic = -2 * ll + 2 * k * jnp.log(n)
    # aicc = aic + (2 * k**2 + 2 * k) / (n - k - 1)
    # return -aicc  # Higher is better.
    return -bic


@jit
def _optimize(y, X, maxiter=500):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta_raw = jnp.ones((p_paths, s_samples))

    def objective(beta_raw, y, X):
        expect = X @ softplus(beta_raw)
        return ((y - expect) ** 2).sum()

    # Estimate beta by minimizing the sum of squared residuals.
    beta_est_raw, opt = jaxopt.LBFGS(Partial(objective, y=y, X=X), maxiter=maxiter).run(
        init_params=init_beta_raw,
    )

    beta_est = softplus(beta_est_raw)
    # Estimate sigma as the root mean sum of squared residuals.
    sigma_est = jnp.sqrt(((y - X @ beta_est) ** 2).mean(0, keepdims=True))
    return (beta_est, sigma_est), opt


def fit(y, X, *args, maxiter=500, **kwargs):
    params_est, opt = _optimize(y=y, X=X, *args, maxiter=maxiter, **kwargs)
    if not opt.iter_num < maxiter:
        raise ConvergenceException(opt)

    # NOTE: Hessian of the *negative* log likelihood, because this is what's being
    # minimized? (How does this make sense??)
    hessian_func = hessian(
        Partial(_negloglik, y=y, X=X, *args, **kwargs), argnums=range(len(params_est))
    )
    # FIXME: Doesn't generalize well to other models. I'll have to change this line.
    beta_est, sigma_est = params_est
    fit_result = FitResult(
        beta_est,
        sigma_est,
        _score(beta_est, sigma_est, y, X),
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
        return fit(y, X, maxiter=self.maxiter)
