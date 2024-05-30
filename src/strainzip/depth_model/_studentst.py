import jax.numpy as jnp
import jaxopt
from jax import hessian, jit
from jax.nn import softplus
from jax.scipy.stats.t import logpdf as StudentsTLogPDF
from jax.tree_util import Partial

from strainzip.errors import ConvergenceException

from ._base import DepthModelResult, JaxDepthModel


def _residual(beta, y, X):
    expect = X @ beta
    return y - expect


@jit
def _fit_studentst_model(y, X, df, maxiter=500):
    e_edges, s_samples = y.shape
    e_edges, p_paths = X.shape
    init_beta_raw = jnp.ones((p_paths, s_samples))
    init_scale_raw = jnp.ones((1,s_samples))

    def objective(params_raw, y, X):
        beta = softplus(params_raw[0])
        scale = softplus(params_raw[1])
        return (-StudentsTLogPDF(_residual(beta, y, X), df = df, loc = 0, scale = scale)).sum()

    # Estimate beta by minimizing the sum of squared residuals.
    (beta_est_raw, scale_est_raw), opt = jaxopt.LBFGS(Partial(objective, y=y, X=X), maxiter=maxiter).run(
        init_params=(init_beta_raw, init_scale_raw), 
    )
    beta_est = softplus(beta_est_raw)
    scale_est = softplus(scale_est_raw)
    # Estimate sigma as the root mean sum of squared residuals.
    #sigma_est = jnp.ones((1,s_samples)) #(jnp.abs(_residual(beta_est, y, X))).mean(0, keepdims=True)
        # NOTE: This has a separate sigma estimate for each sample.
    return dict(beta=beta_est, scale = scale_est), opt


class StudentsTDepthModel(JaxDepthModel):
    param_names = ['scale']

    def __init__(self, maxiter=500, df = 25):
        self.maxiter = maxiter
        self.df = df

    def _fit(self, y, X):
        params, opt = _fit_studentst_model(y=y, X=X, maxiter=self.maxiter, df = self.df)

        if not opt.iter_num < self.maxiter:
            raise ConvergenceException(opt)

        return params, dict(opt=opt)

    def _jax_loglik(self, beta, y, X, **params):
        expect = X @ beta
        return StudentsTLogPDF(y, df = self.df, loc=expect, scale=params['scale']).sum()
