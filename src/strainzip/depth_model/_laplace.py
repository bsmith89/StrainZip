import jax.numpy as jnp
import jaxopt
from jax import hessian, jit
from jax.nn import softplus
from jax.scipy.stats.laplace import logpdf as LaplaceLogPDF
from jax.tree_util import Partial

from strainzip.errors import ConvergenceException

from ._base import DepthModelResult, JaxDepthModel


def _residual(beta, y, X):
    expect = X @ beta
    return y - expect


@jit
def _fit_laplace_model(y, X, maxiter=500):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta_raw = jnp.ones((p_paths, s_samples))

    def objective(beta_raw, y, X):
        beta = softplus(beta_raw)
        return (jnp.abs(_residual(beta, y, X))).sum()

    # Estimate beta by minimizing the sum of squared residuals.
    beta_est_raw, opt = jaxopt.LBFGS(Partial(objective, y=y, X=X), maxiter=maxiter).run(
        init_params=init_beta_raw,
    )
    beta_est = softplus(beta_est_raw)
    # Estimate sigma as the root mean sum of squared residuals.
    sigma_est = (jnp.abs(_residual(beta_est, y, X))).mean(0, keepdims=True)
        # NOTE: This has a separate sigma estimate for each sample.
    return dict(beta=beta_est, sigma=sigma_est), opt


class LaplaceDepthModel(JaxDepthModel):
    param_names = ["sigma"]

    def __init__(self, maxiter=500):
        self.maxiter = maxiter

    def _fit(self, y, X):
        params, opt = _fit_laplace_model(y=y, X=X, maxiter=self.maxiter)

        if not opt.iter_num < self.maxiter:
            raise ConvergenceException(opt)

        return params, dict(opt=opt)

    def _jax_loglik(self, beta, y, X, **params):
        expect = X @ beta
        return LaplaceLogPDF(y, loc=expect, scale=params["sigma"]).sum()
