import jax.numpy as jnp
import jaxopt
from jax import jit
from jax.nn import softplus
from jax.scipy.stats.norm import logpdf as normal_logpdf

from ._base import JaxDepthModel


def _residual(beta, y, X):
    expect = X @ beta
    return y - expect


@jit
def _fit_normal_model(y, X, maxiter, tol):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta_raw = jnp.ones((p_paths, s_samples))

    def objective(beta_raw):
        beta = softplus(beta_raw)
        return (_residual(beta, y, X) ** 2).sum()

    # Estimate beta by minimizing the sum of squared residuals.
    beta_est_raw, opt = jaxopt.LBFGS(objective, maxiter=maxiter, tol=tol).run(
        init_params=init_beta_raw,
    )
    beta_est = softplus(beta_est_raw)
    # Estimate sigma as the root mean sum of squared residuals.
    sigma_est = jnp.sqrt(
        # NOTE: This has a separate sigma estimate for each sample.
        (_residual(beta_est, y, X) ** 2).mean(0, keepdims=True)
    )
    return dict(beta=beta_est, sigma=sigma_est), opt  # beta -> p by s


class NormalDepthModel(JaxDepthModel):
    param_names = ["sigma"]

    def __init__(self, maxiter, tol):
        self.maxiter = maxiter
        self.tol = tol

    def __repr__(self):
        return f"{self.__class__.__name__}(maxiter={self.maxiter}, tol={self.tol})"

    def _fit(self, y, X):
        params, opt = _fit_normal_model(y=y, X=X, maxiter=self.maxiter, tol=self.tol)

        converged = opt.iter_num < self.maxiter

        return params, converged, dict(opt=opt)

    def count_params(self, num_samples, num_edges, num_paths):
        return num_paths * num_samples + num_samples

    def _jax_loglik(self, beta, y, X, **params):
        expect = X @ beta
        return normal_logpdf(y, loc=expect, scale=params["sigma"]).sum()
