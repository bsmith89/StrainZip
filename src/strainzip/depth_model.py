from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jaxopt
import numpy as np
from jax import hessian, jit
from jax.scipy.stats.norm import logpdf as NormalLogPDF
from jax.tree_util import Partial


def _trsfm(x, alpha):
    return jnp.log(x + alpha)


def _inv_trsfm(y, alpha):
    return jnp.exp(y) - alpha


def loglik(beta, sigma, y, X, alpha):
    expect = X @ beta
    trsfm_y = _trsfm(y, alpha=alpha)
    trsfm_expect = _trsfm(expect, alpha=alpha)
    return NormalLogPDF(trsfm_y, loc=trsfm_expect, scale=sigma).sum()


def _negloglik(*args, **kwargs):
    return -1 * loglik(*args, **kwargs)


def _score(beta, sigma, y, X, alpha):
    e_edges, s_samples = y.shape
    _, p_paths = X.shape

    ll = loglik(beta, sigma, y, X, alpha)
    n = e_edges * s_samples
    k = p_paths * s_samples + s_samples
    # NOTE: This parameter count is only correct if fitting both beta and sigma.
    # However, it shouldn't matter for inter-comparison of this same depth model.
    # bic = -2 * ll + 2 * k * jnp.log(n)  # TODO: Consier using BIC instead?
    aic = -2 * ll + 2 * k
    return -aic  # Higher is better.


def _pack_params(beta, sigma, alpha):
    trsfm_beta = _trsfm(beta, alpha)
    log_sigma = jnp.log(sigma)
    return trsfm_beta, log_sigma


def _unpack_params(params, alpha):
    trsfm_beta, log_sigma = params
    beta = _inv_trsfm(trsfm_beta, alpha)
    sigma = jnp.exp(log_sigma)
    return beta, sigma


@jit
def _optimize_beta(init_params, fixed_sigma, y, X, alpha):
    def objective(params, y, X, alpha):
        beta, _ = _unpack_params(params, alpha)
        return -loglik(beta, fixed_sigma, y, X, alpha=alpha)

    opt = jaxopt.LBFGS(Partial(objective, y=y, X=X, alpha=alpha)).run(
        init_params=init_params
    )
    return opt


@jit
def _optimize_sigma(init_params, fixed_beta, y, X, alpha):
    def objective(params, y, X, alpha):
        _, sigma = _unpack_params(params, alpha)
        return -loglik(fixed_beta, sigma, y, X, alpha=alpha)

    opt = jaxopt.LBFGS(Partial(objective, y=y, X=X, alpha=alpha)).run(
        init_params=init_params
    )
    return opt


@dataclass
class FitResult:
    beta: Any
    sigma: Any
    score: float
    hessian_func: Any
    X: Any
    y: Any

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


def fit(y, X, alpha):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta = jnp.ones((p_paths, s_samples))
    init_sigma = jnp.ones((s_samples,))
    init_params = _pack_params(init_beta, init_sigma, alpha)

    opt1 = _optimize_beta(init_params, fixed_sigma=1, y=y, X=X, alpha=alpha)
    est_beta, _ = _unpack_params(opt1.params, alpha)
    opt2 = _optimize_sigma(init_params, fixed_beta=est_beta, y=y, X=X, alpha=alpha)
    _, est_sigma = _unpack_params(opt2.params, alpha=alpha)

    # NOTE: Hessian of the *negative* log likelihood, because this is what's being
    # minimized? (How does this make sense??)
    hessian_func = hessian(Partial(_negloglik, y=y, X=X, alpha=alpha), argnums=[0, 1])
    fit_result = FitResult(
        est_beta,
        est_sigma,
        _score(est_beta, est_sigma, y, X, alpha),
        hessian_func=hessian_func,
        X=X,
        y=y,
    )
    return fit_result
