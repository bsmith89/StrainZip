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
    init_scale_raw = jnp.ones((1, s_samples))

    def objective(params_raw):
        beta_raw, scale_raw = params_raw
        beta = softplus(beta_raw)
        scale = softplus(scale_raw)
        return -laplace_logpdf(_residual(beta, y, X), loc=0, scale=scale).sum()

    # Estimate beta by minimizing the sum of squared residuals.
    (beta_est_raw, scale_est_raw), opt = jaxopt.LBFGS(objective, maxiter=maxiter).run(
        init_params=(init_beta_raw, init_scale_raw)
    )
    beta_est = softplus(beta_est_raw)
    scale_est = softplus(scale_est_raw)
    return dict(beta=beta_est, scale=scale_est), opt


class LaplaceDepthModel(JaxDepthModel):
    param_names = ["scale"]

    def __init__(self, maxiter=500):
        self.maxiter = maxiter

    def _fit(self, y, X):
        params, opt = _fit_laplace_model(y=y, X=X, maxiter=self.maxiter)

        converged = opt.iter_num < self.maxiter

        return params, converged, dict(opt=opt)

    def count_params(self, num_samples, num_edges, num_paths):
        return num_paths * num_samples + num_samples

    def _jax_loglik(self, beta, y, X, **params):
        expect = X @ beta
        return laplace_logpdf(y, loc=expect, scale=params["scale"]).sum()
