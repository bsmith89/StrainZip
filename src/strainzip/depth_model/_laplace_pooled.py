import jax.numpy as jnp
import jaxopt
from jax import jit
from jax.nn import softplus
from jax.scipy.stats.laplace import logpdf as laplace_logpdf

from ._base import JaxDepthModel


def _residual(beta, y, X):
    expect = X @ beta
    return y - expect


@jit
def _fit_laplace_model(y, X, maxiter=500):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta_raw = jnp.ones((p_paths, s_samples))
    init_scale_raw = jnp.ones((1, 1))

    def objective1(beta_raw):
        beta = softplus(beta_raw)
        return -laplace_logpdf(_residual(beta, y, X), loc=0, scale=1).sum()

    # Estimate beta by minimizing the sum of squared residuals.
    beta_est_raw, opt1 = jaxopt.LBFGS(objective1, maxiter=maxiter).run(
        init_params=init_beta_raw
    )
    beta_est = softplus(beta_est_raw)

    def objective2(scale_raw):
        scale = softplus(scale_raw)
        return -laplace_logpdf(_residual(beta_est, y, X), loc=0, scale=scale).sum()

    # Estimate beta by minimizing the sum of squared residuals.
    scale_est_raw, opt2 = jaxopt.LBFGS(objective2, maxiter=maxiter).run(
        init_params=init_scale_raw
    )

    beta_est = softplus(beta_est_raw)
    scale_est = softplus(scale_est_raw)
    return dict(beta=beta_est, scale=scale_est), opt1, opt2


class LaplacePooledDepthModel(JaxDepthModel):
    param_names = ["scale"]

    def __init__(self, maxiter=500):
        self.maxiter = maxiter

    def _fit(self, y, X):
        params, opt1, opt2 = _fit_laplace_model(y=y, X=X, maxiter=self.maxiter)

        converged = (opt1.iter_num < self.maxiter) & (opt2.iter_num < self.maxiter)

        return params, converged, dict(opt1=opt1, opt2=opt2)

    def count_params(self, num_samples, num_edges, num_paths):
        return num_paths * num_samples + 1

    def _jax_loglik(self, beta, y, X, **params):
        expect = X @ beta
        return laplace_logpdf(y, loc=expect, scale=params["scale"]).sum()
