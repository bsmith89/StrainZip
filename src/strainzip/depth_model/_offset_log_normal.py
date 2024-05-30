import jax.numpy as jnp
import jaxopt
from jax import hessian, jit
from jax.scipy.stats.norm import logpdf as NormalLogPDF
from jax.tree_util import Partial

from strainzip.errors import ConvergenceException

from ._base import BaseDepthModel, DepthModelResult


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
        ((_trsfm(y, alpha) - _trsfm(X @ beta_est, alpha)) ** 2).mean(0, keepdims=True)
        # # NOTE (2024-05-17): Experimenting with a variance pooled across samples.
        # ((_trsfm(y, alpha) - _trsfm(X @ beta_est, alpha)) ** 2).mean(keepdims=True)
    )
    return (beta_est, sigma_est), opt


def fit(y, X, alpha, maxiter=500):
    params_est, opt = _optimize(y=y, X=X, alpha=alpha, maxiter=maxiter)
    if not opt.iter_num < maxiter:
        raise ConvergenceException(opt)

    # NOTE: Doesn't generalize well to other models. I'll have to change this line.
    beta_est, sigma_est = params_est
    return beta_est, sigma_est, opt


class OffsetLogNormalDepthModel(BaseDepthModel):
    def __init__(self, alpha, maxiter=500):
        self.alpha = alpha
        self.maxiter = maxiter

    def fit(self, y, X):
        beta_est, sigma_est, opt = fit(y, X, self.alpha, self.maxiter)
        return DepthModelResult(
            model=self,
            params=dict(beta=beta_est, sigma=sigma_est),
            X=X,
            y=y,
            debug=dict(opt=opt),
        )

    def loglik(self, beta, y, X, **kwargs):
        return loglik(beta, kwargs["sigma"], y, X, self.alpha)

    # NOTE: Hessian of the *negative* log likelihood, because this is what's being
    # minimized? (How does this make sense??)
    def hessian(self, beta, y, X, **kwargs):
        return hessian(Partial(_negloglik, y=y, X=X, alpha=self.alpha), argnums=[0, 1])(
            beta, kwargs["sigma"]
        )
