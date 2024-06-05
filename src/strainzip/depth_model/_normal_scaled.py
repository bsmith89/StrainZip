import jax.numpy as jnp
import jaxopt
from jax import jit
from jax.nn import softplus
from jax.scipy.stats.norm import logpdf as normal_logpdf
from jax.tree_util import Partial

from ._base import JaxDepthModel


def _residual(beta, y, X):
    expect = X @ beta
    return y - expect


@jit
def _fit_normal_scaled_model(y, X, alpha, maxiter, tol):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape

    def objective1(beta_raw, sigma_raw=None):
        beta = softplus(beta_raw)
        sigma = softplus(sigma_raw)
        expect = X @ beta
        return -normal_logpdf(y, loc=expect, scale=sigma * expect**alpha).sum()

    def objective2(sigma_raw, beta_raw=None):
        beta = softplus(beta_raw)
        sigma = softplus(sigma_raw)
        expect = X @ beta
        return -normal_logpdf(y, loc=expect, scale=sigma * expect**alpha).sum()

    # Initialize beta and sigma estimates.
    beta_est_raw = jnp.ones((p_paths, s_samples))
    sigma_est_raw = jnp.ones((1, 1))
    opt1 = opt2 = None  # Just to satisfy pyright
    # Estimate beta by minimizing the sum of squared residuals.
    # FIXME: Up the number of iterations of this alternation.
    for i in range(2):
        beta_est_raw, opt1 = jaxopt.LBFGS(
            Partial(objective1, sigma_raw=sigma_est_raw), maxiter=maxiter, tol=tol
        ).run(
            init_params=beta_est_raw,
        )
        sigma_est_raw, opt2 = jaxopt.LBFGS(
            Partial(objective2, beta_raw=beta_est_raw), maxiter=maxiter, tol=tol
        ).run(
            init_params=sigma_est_raw,
        )

    beta_est = softplus(beta_est_raw)
    sigma_est = softplus(sigma_est_raw)

    # FIXME: This is only the most recent opt value...
    return dict(beta=beta_est, sigma=sigma_est), (opt1, opt2)


class NormalScaledDepthModel(JaxDepthModel):
    param_names = ["sigma"]

    def __init__(self, alpha, maxiter, tol):
        self.alpha = alpha
        self.maxiter = maxiter
        self.tol = tol

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, maxiter={self.maxiter}, tol={self.tol})"

    def _fit(self, y, X):
        params, (opt1, opt2) = _fit_normal_scaled_model(
            y=y, X=X, alpha=self.alpha, maxiter=self.maxiter, tol=self.tol
        )

        converged = (opt1.iter_num < self.maxiter) & (opt2.iter_num < self.maxiter)

        return params, converged, dict(opt1=opt1, opt2=opt2)

    def count_params(self, num_samples, num_edges, num_paths):
        return num_paths * num_samples + 1

    def _jax_loglik(self, beta, y, X, **params):
        expect = X @ beta
        return normal_logpdf(
            y, loc=expect, scale=params["sigma"] * expect**self.alpha
        ).sum()
