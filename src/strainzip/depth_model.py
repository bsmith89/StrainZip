from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jaxopt
import numpy as np
from jax import hessian, jit
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
        return loglik(self.beta, self.sigma, self.y, self.X, self.alpha)

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
        # # FIXME (2024-05-17): Experimenting with a global variance.
        # return self.num_paths * self.num_samples + self.num_samples
        return self.num_paths * self.num_samples + 1

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
def _optimize(y, X, alpha, maxiter=500):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta = jnp.ones((p_paths, s_samples))

    def objective(beta_trsfm, y, X, alpha):
        return (_residual(_inv_trsfm(beta_trsfm, alpha), y, X, alpha) ** 2).sum()

    # low_bounds = 0. * jnp.ones_like(init_beta)
    # upp_bounds = jnp.inf * jnp.ones_like(init_beta)

    # Estimate beta by minimizing the sum of squared residuals.
    beta_est_trsfm, opt = jaxopt.LBFGS(
        Partial(objective, y=y, X=X, alpha=alpha), maxiter=maxiter
    ).run(
        init_params=init_beta,
        # bounds=(low_bounds, upp_bounds),
    )
    beta_est = _inv_trsfm(beta_est_trsfm, alpha)
    # Estimate sigma as the root mean sum of squared residuals.
    sigma_est = jnp.sqrt(
        # # FIXME (2024-05-17): Experimenting with a variance pooled across samples.
        # ((_trsfm(y, alpha) - _trsfm(X @ beta_est, alpha)) ** 2).mean(0, keepdims=True)
        ((_trsfm(y, alpha) - _trsfm(X @ beta_est, alpha)) ** 2).mean(keepdims=True)
    )
    return (beta_est, sigma_est), opt


def fit(y, X, alpha, maxiter=500):
    params_est, opt = _optimize(y=y, X=X, alpha=alpha, maxiter=maxiter)
    if not opt.iter_num < maxiter:
        raise ConvergenceException(opt)

    # NOTE: Hessian of the *negative* log likelihood, because this is what's being
    # minimized? (How does this make sense??)
    hessian_func = hessian(
        Partial(_negloglik, y=y, X=X, alpha=alpha), argnums=range(len(params_est))
    )
    # NOTE: Doesn't generalize well to other models. I'll have to change this line.
    beta_est, sigma_est = params_est
    fit_result = FitResult(
        beta=beta_est,
        sigma=sigma_est,
        hessian_func=hessian_func,
        X=X,
        y=y,
        alpha=alpha,
        opt=opt,
    )
    return fit_result


class LogPlusAlphaLogNormal:
    def __init__(self, alpha, maxiter=500):
        self.alpha = alpha
        self.maxiter = maxiter

    def fit(self, y, X):
        return fit(y, X, self.alpha, self.maxiter)
