import jax.numpy as jnp
import jaxopt
from jax import grad, jit
from jax.nn import softplus
from jax.scipy.stats.norm import logpdf as normal_logpdf
from jax.tree_util import Partial
from scipy.optimize import nnls

from ._base import JaxDepthModel


def weighted_non_negative_least_squares(y, X, W, maxiter=None, atol=None):
    # Based on https://stackoverflow.com/a/36112536
    sqrtW = jnp.sqrt(W)
    beta, _ = nnls(sqrtW @ X, y @ sqrtW, maxiter=maxiter, atol=atol)
    return beta


# @jit
def multi_sample_weighted_least_squares(Y, X, W, maxiter=None, atol=None):
    beta = []
    for i in range(Y.shape[1]):
        sample_W = jnp.diag(W[:, i])
        sample_y = Y[:, i]
        beta.append(
            weighted_non_negative_least_squares(
                sample_y, X, sample_W, maxiter=maxiter, atol=atol
            )
        )
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


def step_alternating_reweighted_least_squares(beta, Y, X, alpha):
    sigma = estimate_sigma(beta, Y, X, alpha)
    W = 1 / calculate_expected_var(beta, X, sigma, alpha)
    beta = multi_sample_weighted_least_squares(Y, X, W)
    return beta, sigma


# def iterate_alternating_reweighted_least_squares(Y, X, alpha):
#     W_t = jnp.ones_like(Y)
#     while True:
#         beta_t = multi_sample_weighted_least_squares(Y, X, W_t)
#         sigma_t = estimate_sigma(beta_t, Y, X, alpha)
#         W_t = calculate_expected_var(beta_t, X, sigma_t, alpha)
#         yield beta_t, sigma_t
#


# @jit
def loglik(beta, sigma, Y, X, alpha):
    expect = X @ beta
    return normal_logpdf(Y, loc=expect, scale=sigma * expect**alpha).sum()


loglik_grad_func = grad(loglik, argnums=[0, 1])


def _fit_normal_scaled_model(y, X, alpha, maxiter, tol):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape

    converged = False
    grad_norm = jnp.inf
    beta_est = jnp.ones((p_paths, s_samples))
    sigma_est = jnp.ones((1, 1))
    beta_grad = jnp.nan * beta_est
    sigma_grad = jnp.nan * sigma_est
    i = -1
    for i in range(maxiter):
        beta_est, sigma_est = step_alternating_reweighted_least_squares(
            beta_est, y, X, alpha
        )
        beta_grad, sigma_grad = loglik_grad_func(beta_est, sigma_est, y, X, alpha)
        beta_grad_ss = (beta_grad**2).sum()
        sigma_grad_ss = (sigma_grad**2).sum()
        grad_norm = (beta_grad_ss + sigma_grad_ss) ** (1 / 2)

        if grad_norm == jnp.nan:
            break
        if grad_norm < tol:
            converged = True
            break

    return dict(beta=beta_est, sigma=sigma_est), {
        "converged": converged,
        "grad_norm": grad_norm,
        "beta_grad": beta_grad,
        "sigma_grad": sigma_grad,
        "iter_num": i,
    }


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
