import jax.numpy as jnp
import jaxopt
import numpy as np
from jax import hessian, jit
from jax.scipy.stats.norm import logpdf as NormalLogPDF
from jax.tree_util import Partial


def _stable_log(x, alpha=1e-5):
    return jnp.log(x + alpha)


def _inv_stable_log(x, alpha=1e-5):
    return jnp.exp(x) - alpha


def negloglik(beta, sigma, y, X, alpha=1e-5):
    expect = X @ beta
    slog_y = _stable_log(y, alpha=alpha)
    slog_expect = _stable_log(expect, alpha=alpha)
    return -NormalLogPDF(slog_y, loc=slog_expect, scale=sigma).sum()


def _packed_unbounded_negloglik(params, y, X, alpha=1e-5):
    slog_beta, log_sigma = params
    beta = _inv_stable_log(slog_beta, alpha=alpha)
    sigma = jnp.exp(log_sigma)
    return negloglik(beta, sigma, y, X, alpha=alpha)


@jit
def fit(y, X, alpha=1e-5):
    slog_beta_init = jnp.zeros((X.shape[1], y.shape[1]))
    log_sigma_init = jnp.zeros((1,))
    fit = jaxopt.LBFGS(Partial(_packed_unbounded_negloglik, y=y, X=X, alpha=alpha)).run(
        init_params=(slog_beta_init, log_sigma_init)
    )
    slog_beta, log_sigma = fit.params
    return _inv_stable_log(slog_beta, alpha), jnp.exp(log_sigma), fit


def estimate_stderr(y, X, beta_est, sigma_est, alpha=1e-5):
    p_paths, s_samples = beta_est.shape
    model_hessian = hessian(Partial(negloglik, y=y, X=X, alpha=alpha), argnums=[0, 1])
    (beta_beta_hess, beta_sigma_hess), (
        sigm_beta_hess,
        sigma_sigma_hess,
    ) = model_hessian(beta_est, sigma_est)
    beta_hess_flat = beta_beta_hess.reshape((p_paths * s_samples, -1))
    beta_var_covar_matrix = jnp.linalg.inv(beta_hess_flat)
    beta_variance = np.diag(beta_var_covar_matrix).reshape((p_paths, s_samples))
    beta_stderr = np.sqrt(beta_variance)

    sigma_variance = jnp.linalg.inv(sigma_sigma_hess)
    sigma_stderr = np.sqrt(sigma_variance)

    return beta_stderr, sigma_stderr, beta_var_covar_matrix
