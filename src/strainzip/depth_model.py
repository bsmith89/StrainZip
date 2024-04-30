from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jaxopt
import numpy as np
from jax import hessian, jit
from jax.scipy.stats.norm import logpdf as NormalLogPDF
from jax.tree_util import Partial


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


def _trsfm(x, alpha):
    return jnp.log(x + alpha)


def _inv_trsfm(y, alpha):
    return jnp.exp(y) - alpha


def loglik(beta, sigma, y, X, alpha):
    expect = X @ beta
    y_trsfm = _trsfm(y, alpha=alpha)
    expect_trsfm = _trsfm(expect, alpha=alpha)
    return NormalLogPDF(y_trsfm, loc=expect_trsfm, scale=sigma).sum()


def _negloglik(*args, **kwargs):
    return -1 * loglik(*args, **kwargs)


def _residual(beta, y, X, alpha):
    expect = X @ beta
    y_trsfm = _trsfm(y, alpha=alpha)
    expect_trsfm = _trsfm(expect, alpha=alpha)
    return y_trsfm - expect_trsfm


def _score(beta, sigma, y, X, alpha):
    e_edges, s_samples = y.shape
    _, p_paths = X.shape

    ll = loglik(beta, sigma, y, X, alpha)
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


# def _pack_params(beta, sigma, alpha):
#     trsfm_beta = _trsfm(beta, alpha)
#     log_sigma = jnp.log(sigma)
#     return trsfm_beta, log_sigma
#
#
# def _unpack_params(params, alpha):
#     trsfm_beta, log_sigma = params
#     beta = _inv_trsfm(trsfm_beta, alpha)
#     sigma = jnp.exp(log_sigma)
#     return beta, sigma


@jit
def _optimize(y, X, alpha):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta = jnp.ones((p_paths, s_samples))

    def objective(beta_trsfm, y, X, alpha):
        return (_residual(_inv_trsfm(beta_trsfm, alpha), y, X, alpha) ** 2).sum()

    # low_bounds = 0. * jnp.ones_like(init_beta)
    # upp_bounds = jnp.inf * jnp.ones_like(init_beta)

    # Estimate beta by minimizing the sum of squared residuals.
    beta_est_trsfm, opt = jaxopt.LBFGS(Partial(objective, y=y, X=X, alpha=alpha)).run(
        init_params=init_beta,
        # bounds=(low_bounds, upp_bounds),
    )
    beta_est = _inv_trsfm(beta_est_trsfm, alpha)
    # Estimate sigma as the root mean sum of squared residuals.
    sigma_est = jnp.sqrt(
        ((_trsfm(y, alpha) - _trsfm(X @ beta_est, alpha)) ** 2).mean(1, keepdims=True)
    )
    return (beta_est, sigma_est), opt


def fit(y, X, alpha):
    params_est, opt = _optimize(y=y, X=X, alpha=alpha)

    # NOTE: Hessian of the *negative* log likelihood, because this is what's being
    # minimized? (How does this make sense??)
    hessian_func = hessian(
        Partial(_negloglik, y=y, X=X, alpha=alpha), argnums=range(len(params_est))
    )
    # FIXME: Doesn't generalize well to other models. I'll have to change this line.
    beta_est, sigma_est = params_est
    fit_result = FitResult(
        beta_est,
        sigma_est,
        _score(beta_est, sigma_est, y, X, alpha=alpha),
        hessian_func=hessian_func,
        X=X,
        y=y,
        opt=opt,
    )
    return fit_result


class LogPlusAlphaLogNormal:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, y, X):
        return fit(y, X, self.alpha)
