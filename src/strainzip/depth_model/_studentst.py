import jax.numpy as jnp
import jaxopt
from jax import jit
from jax.nn import softplus
from jax.scipy.stats.t import logpdf as studentst_logpdf

from ._base import JaxDepthModel


def _residual(beta, y, X):
    expect = X @ beta
    return y - expect


@jit
def _fit_studentst_model(y, X, df, maxiter=500):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta_raw = jnp.ones((p_paths, s_samples))
    init_scale_raw = jnp.ones((1, s_samples))

    def objective1(beta_raw):
        beta = softplus(beta_raw)
        return (_residual(beta, y, X) ** 2).sum()

    # Estimate beta by minimizing the sum of squared residuals.
    beta_est_raw, opt1 = jaxopt.LBFGS(objective1, maxiter=maxiter).run(
        init_params=init_beta_raw,
    )
    beta_est = softplus(beta_est_raw)

    residuals = _residual(beta_est, y, X)

    def objective2(scale_raw):
        scale = softplus(scale_raw)
        return -studentst_logpdf(residuals, df=df, loc=0, scale=scale).sum()

    scale_est_raw, opt2 = jaxopt.LBFGS(objective2, maxiter=maxiter).run(
        init_params=init_scale_raw,
    )
    scale_est = softplus(scale_est_raw)

    # NOTE: This has a separate sigma estimate for each sample.
    return dict(beta=beta_est, scale=scale_est), opt1, opt2


class StudentsTDepthModel(JaxDepthModel):
    param_names = ["scale"]

    def __init__(self, df, maxiter=500):
        self.maxiter = maxiter
        self.df = df

    def _fit(self, y, X):
        params, opt1, opt2 = _fit_studentst_model(
            y=y, X=X, maxiter=self.maxiter, df=self.df
        )

        converged = (opt1.iter_num < self.maxiter) & (opt2.iter_num < self.maxiter)

        return params, converged, dict(opt1=opt1, opt2=opt2)

    def count_params(self, num_samples, num_edges, num_paths):
        return num_paths * num_samples + num_samples

    def _jax_loglik(self, beta, y, X, **params):
        expect = X @ beta
        return studentst_logpdf(y, df=self.df, loc=expect, scale=params["scale"]).sum()
