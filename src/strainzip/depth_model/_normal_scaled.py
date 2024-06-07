import jax.numpy as jnp
import jaxopt
from jax import grad, jit
from jax.nn import softplus
from jax.scipy.stats.norm import logpdf as normal_logpdf
from jax.tree_util import Partial

from ._base import JaxDepthModel


def weighted_least_squares(y, X, W):
    beta = jnp.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    return beta


def residual(beta, Y, X):
    expect = X @ beta
    resid = Y - expect
    return resid


def multi_sample_weighted_least_squares(Y, X, W):
    beta = []
    for i in range(Y.shape[1]):
        sample_W = jnp.diag(W[:, i])
        sample_y = Y[:, i]
        beta.append(weighted_least_squares(sample_y, X, sample_W))
    return jnp.stack(beta, axis=1)


def estimate_sigma(beta, Y, X, alpha):
    expect = X @ beta
    expect_trsfm = expect**alpha
    resid = Y - expect
    rescaled_resid = resid / expect_trsfm
    rescaled_rmse = (rescaled_resid**2).mean() ** (1 / 2)
    return rescaled_rmse


def calculate_expected_var(beta, X, sigma, alpha):
    expect = X @ beta
    expect_var = (sigma * expect**alpha) ** 2
    return expect_var


def iterate_alternating_reweighted_least_squares(Y, X, alpha):
    W_t = jnp.ones_like(Y)
    while True:
        beta_t = multi_sample_weighted_least_squares(Y, X, W=W_t)
        sigma_t = estimate_sigma(beta_t, Y, X, alpha)
        W_t = calculate_expected_var(beta_t, X, sigma_t, alpha)
        yield beta_t, sigma_t


def loglik(beta, sigma, Y, X, alpha):
    expect = X @ beta
    return normal_logpdf(Y, loc=expect, scale=sigma * expect**alpha).sum()


@jit
def _fit_normal_scaled_model(y, X, alpha, maxiter, tol):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape

    loglik_grad_func = grad(loglik)

    converged = False
    grad_norm = jnp.inf
    beta_est = jnp.nan * jnp.ones((p_paths, s_samples))
    sigma_est = jnp.nan
    i = -1
    for i, (beta_est, sigma_est) in enumerate(
        iterate_alternating_reweighted_least_squares(y, X, alpha)
    ):
        grad_norm = (loglik_grad_func(beta_est, sigma_est, y, X, alpha) ** 2).sum() ** (
            1 / 2
        )
        if grad_norm == jnp.nan:
            break
        if grad_norm < tol:
            converged = True
            break
        if i > maxiter:
            break

    return dict(beta=beta_est, sigma=sigma_est), dict(
        converged=converged, grad_norm=grad_norm, iter_num=i
    )


class NormalScaledDepthModel(JaxDepthModel):
    param_names = ["sigmaB"]

    def __init__(self, alpha, maxiter, tol):
        self.alpha = alpha
        self.maxiter = maxiter
        self.tol = tol

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, maxiter={self.maxiter}, tol={self.tol})"

    def _fit(self, y, X):
        params, opt = _fit_normal_scaled_model(
            y=y, X=X, alpha=self.alpha, maxiter=self.maxiter, tol=self.tol
        )

        return params, opt["converged"], dict(opt=opt)

    def count_params(self, num_samples, num_edges, num_paths):
        return num_paths * num_samples + 1

    def _jax_loglik(self, beta, y, X, **params):
        return loglik(beta, params["sigma"], y, X, self.alpha)