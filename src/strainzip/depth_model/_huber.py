import jax.numpy as jnp
import jaxopt
from jax import jit, vmap
from jax.nn import softplus
from jax.scipy.stats.norm import logpdf as laplace_logpdf
from jax.tree_util import Partial
from jaxopt.loss import huber_loss

from ._base import JaxDepthModel


def _residual(beta, y, X):
    expect = X @ beta
    return y - expect


@jit
def _fit_huber_model(y, X, delta, maxiter=500):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta_raw = jnp.ones((p_paths, s_samples))

    def objective(beta_raw):
        beta = softplus(beta_raw)
        elementwise_huber_norm = vmap(Partial(huber_loss, pred=0, delta=delta))
        return jnp.sum(elementwise_huber_norm(_residual(beta, y, X)))

    # Estimate beta by minimizing the Huber loss
    beta_est_raw, opt = jaxopt.LBFGS(objective, maxiter=maxiter).run(
        init_params=init_beta_raw,
    )
    beta_est = softplus(beta_est_raw)
    return dict(beta=beta_est), opt


class HuberDepthModel(JaxDepthModel):
    param_names = []

    def __init__(self, delta, maxiter=500):
        self.maxiter = maxiter
        self.delta = delta

    def _fit(self, y, X):
        params, opt = _fit_huber_model(y=y, X=X, delta=self.delta, maxiter=self.maxiter)

        converged = opt.iter_num < self.maxiter

        return params, converged, dict(opt=opt)

    def count_params(self, num_samples, num_edges, num_paths):
        return num_paths * num_samples

    def _jax_loglik(self, beta, y, X, **params):
        expect = X @ beta
        # FIXME (2024-05-31): Laplace LL should be replaced with Huber density
        return laplace_logpdf(y, loc=expect, scale=1).sum()
