import jax.numpy as jnp
import jaxopt
from jax import jit
from jax.scipy.stats.norm import logpdf as normal_logpdf

from ._base import JaxDepthModel


def _trsfm(x, alpha):
    return jnp.log(x + alpha)


def _inv_trsfm(y, alpha):
    return jnp.exp(y) - alpha


def _residual(beta, y, X, alpha):
    expect = X @ beta
    y_trsfm = _trsfm(y, alpha=alpha)
    expect_trsfm = _trsfm(expect, alpha=alpha)
    return y_trsfm - expect_trsfm


@jit
def _fit_offset_log_normal_model(y, X, alpha, maxiter=500):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta = jnp.ones((p_paths, s_samples))

    def objective(beta_trsfm):
        return (_residual(_inv_trsfm(beta_trsfm, alpha), y, X, alpha) ** 2).sum()

    # Estimate beta by minimizing the sum of squared residuals.
    beta_est_trsfm, opt = jaxopt.LBFGS(objective, maxiter=maxiter).run(
        init_params=init_beta,
    )
    beta_est = _inv_trsfm(beta_est_trsfm, alpha)
    # Estimate sigma as the root mean sum of squared residuals.
    sigma_est = jnp.sqrt(
        ((_trsfm(y, alpha) - _trsfm(X @ beta_est, alpha)) ** 2).mean(0, keepdims=True)
        # # NOTE (2024-05-17): Experimenting with a variance pooled across samples.
        # ((_trsfm(y, alpha) - _trsfm(X @ beta_est, alpha)) ** 2).mean(keepdims=True)
    )
    return dict(beta=beta_est, sigma=sigma_est), opt


class OffsetLogNormalDepthModel(JaxDepthModel):
    param_names = ["sigma"]

    def __init__(self, alpha, maxiter=500):
        self.alpha = alpha
        self.maxiter = maxiter

    def _fit(self, y, X):
        params, opt = _fit_offset_log_normal_model(
            y=y, X=X, alpha=self.alpha, maxiter=self.maxiter
        )

        converged = opt.iter_num < self.maxiter

        return params, converged, dict(opt=opt)

    def count_params(self, num_samples, num_edges, num_paths):
        return num_paths * num_samples + num_samples

    def _jax_loglik(self, beta, y, X, **params):
        expect = X @ beta
        y_trsfm = _trsfm(y, alpha=self.alpha)
        expect_trsfm = _trsfm(expect, alpha=self.alpha)
        return normal_logpdf(y_trsfm, loc=expect_trsfm, scale=params["sigma"]).sum()
