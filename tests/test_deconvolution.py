import numpy as np

import strainzip as sz


def test_deconvolution_problem_formulation():
    X, y, labels = sz.deconvolution.formulate_path_deconvolution(
        np.array([[100, 20, 50], [0, 0, 0]]),
        np.array([[100, 10, 0], [0, 10, 50], [0, 0, 0]]),
    )
    assert np.array_equal(
        X,
        [
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ],
    )
    assert np.array_equal(
        y,
        [
            [100, 20, 50],
            [0, 0, 0],
            [100, 10, 0],
            [0, 10, 50],
            [0, 0, 0],
        ],
    )


def test_well_specified_deconvolution():
    seed = 0
    alpha = 1e-5  # Small offset for handling 0s in depths
    n, m = 2, 3  # In-edges / out-edges
    s_samples = 3
    sigma = 1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    np.random.seed(seed)

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_paths(n, m)[0]
    assert X.shape == (r_edges, p_paths)

    # Select which pairs of in/out edges are "real" and assign them weights across samples.
    active_paths = sz.deconvolution.simulate_active_paths(n, m)
    active_paths = [i for i, _ in active_paths]
    beta = np.zeros((p_paths, s_samples))
    beta[active_paths, :] = np.random.lognormal(
        mean=-3, sigma=6, size=(len(active_paths), s_samples)
    )
    beta = beta.round(1)  # Structural zeros

    # Simulate the observed depth of each edge.
    expect = X @ (beta * depth_multiplier)
    log_noise = np.random.normal(loc=0, scale=1, size=expect.shape)
    y_obs = expect * np.exp(log_noise * sigma)

    # Simulate a selection of paths during the estimation procedure.
    # Possibly over-specified. (see `num_excess_paths`)
    _active_paths = list(
        sorted(
            set(active_paths)
            | set(
                np.random.choice(
                    [p for p in range(p_paths) if p not in active_paths],
                    replace=False,
                    size=num_excess_paths,
                )
            )
        )
    )
    X_reduced = X[:, _active_paths]

    # Estimate model parameters
    beta_est, sigma_est, _ = sz.depth_model.fit(y_obs, X_reduced, alpha=alpha)

    # Calculate likelihood
    loglik = -sz.depth_model.negloglik(
        beta_est, sigma_est, y_obs, X_reduced, alpha=alpha
    )
    assert np.isfinite(loglik)

    # Estimate standard errors.
    beta_stderr, sigma_stderr, inv_beta_hessian = sz.depth_model.estimate_stderr(
        y_obs,
        X_reduced,
        beta_est,
        sigma_est,
        alpha=alpha,
    )

    # Check model identifiable.
    assert np.isfinite(beta_stderr).all()
    assert np.isfinite(sigma_stderr).all()


def test_predefined_deconvolution():
    alpha = 1e-5  # Small offset for handling 0s in depths
    n, m = 2, 3  # In-edges / out-edges
    s_samples = 3
    sigma = 1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_paths(n, m)[0]
    assert X.shape == (r_edges, p_paths)

    # Select which pairs of in/out edges are "real" and assign them weights across samples.
    beta = np.array(
        [
            [0.0000e00, 0.0000e00, 0.0000e00],
            [4.3000e00, 5.5420e02, 0.0000e00],
            [1.4860e02, 0.0000e00, 6.7652e03],
            [9.9200e01, 0.0000e00, 2.1346e05],
            [0.0000e00, 0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00, 0.0000e00],
        ]
    )

    # Specify the observed depth of each edge.
    expect = X @ (beta * depth_multiplier)
    log_noise = np.array(
        [
            [1.08081191, 0.48431215, 0.57914048],
            [-0.18158257, 1.41020463, -0.37447169],
            [0.27519832, -0.96075461, 0.37692697],
            [0.03343893, 0.68056724, -1.56349669],
            [-0.56669762, -0.24214951, 1.51439128],
        ]
    )
    y_obs = expect * np.exp(log_noise * sigma)

    # Select included paths during the estimation procedure.
    # Possibly over-specified. (see `num_excess_paths`)
    _active_paths = [1, 2, 3]
    X_reduced = X[:, _active_paths]

    # Estimate model parameters
    beta_reduced_est, sigma_est, _ = sz.depth_model.fit(y_obs, X_reduced, alpha=alpha)
    # Check estimates.
    assert np.allclose(
        beta_reduced_est,
        np.array(
            [
                [4.5332198e00, 9.9224072e02, -8.1854523e-11],
                [1.9077873e02, 3.6379788e-12, 1.9270121e04],
                [1.0395216e02, 3.6379788e-12, 2.1372383e05],
            ]
        ),
    )
    assert np.allclose(sigma_est, np.array([0.5422841, 0.06206167, 0.379378]))

    # Check likelihood
    loglik = -sz.depth_model.negloglik(
        beta_reduced_est, sigma_est, y_obs, X_reduced, alpha=alpha
    )
    assert np.allclose(loglik, 0.52001405)

    # Estimate standard errors.
    (
        beta_reduced_stderr,
        sigma_stderr,
        inv_beta_hessian,
    ) = sz.depth_model.estimate_stderr(
        y_obs,
        X_reduced,
        beta_reduced_est,
        sigma_est,
        alpha=alpha,
    )
    # Check estimates.
    assert np.allclose(
        beta_reduced_stderr,
        np.array(
            [
                [2.4823506e00, 4.3543732e01, 3.7937325e-06],
                [7.4403160e01, 6.2061696e-07, 5.1694121e03],
                [3.9860443e01, 4.3884245e-07, 5.7333934e04],
            ]
        ),
    )
    assert np.allclose(sigma_stderr, np.array([0.17148595, 0.01962585, 0.11996878]))


def test_model_selection_procedure_3x4():
    seed = 0
    alpha = 1e-0  # Small offset for handling 0s in depths
    n, m = 3, 4  # In-edges / out-edges
    s_samples = 4
    sigma = 1e-1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 1  # How many extra paths to include beyond correct ones.

    np.random.seed(seed)

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_paths(n, m)[0]
    assert X.shape == (r_edges, p_paths)

    # Select which pairs of in/out edges are "real" and assign them weights across samples.
    active_paths = sz.deconvolution.simulate_active_paths(n, m, excess=num_excess_paths)
    active_paths = [i for i, _ in active_paths]
    beta = np.zeros((p_paths, s_samples))
    beta[active_paths, :] = np.random.lognormal(
        mean=-5, sigma=7, size=(len(active_paths), s_samples)
    )
    beta = beta.round(1)  # Structural zeros

    # Simulate the observed depth of each edge.
    expect = X @ (beta * depth_multiplier)
    log_noise = np.random.normal(loc=0, scale=1, size=expect.shape)
    y_obs = expect * np.exp(log_noise * sigma)

    # Select paths and estimate depth
    (
        selected_paths,
        beta_est,
        beta_stderr,
        sigma_est,
        sigma_stderr,
        inv_hessian,
        fit,
        delta_aic,
    ) = sz.deconvolution.estimate_paths(
        X,
        y_obs,
        model=sz.depth_model,
        forward_stop=0.2,
        backward_stop=0.01,
        alpha=alpha,
    )

    assert set(selected_paths) == set(active_paths)


# FIXME: Parameterize the previous test instead of making this nearly identical test.
def test_model_selection_procedure_2x1():
    seed = 0
    alpha = 1e-0  # Small offset for handling 0s in depths
    n, m = 2, 1  # In-edges / out-edges
    s_samples = 4
    sigma = 1e-1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    np.random.seed(seed)

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_paths(n, m)[0]
    assert X.shape == (r_edges, p_paths)

    # Select which pairs of in/out edges are "real" and assign them weights across samples.
    active_paths = sz.deconvolution.simulate_active_paths(n, m, excess=num_excess_paths)
    active_paths = [i for i, _ in active_paths]
    beta = np.zeros((p_paths, s_samples))
    beta[active_paths, :] = np.random.lognormal(
        mean=-5, sigma=7, size=(len(active_paths), s_samples)
    )
    beta = beta.round(1)  # Structural zeros

    # Simulate the observed depth of each edge.
    expect = X @ (beta * depth_multiplier)
    log_noise = np.random.normal(loc=0, scale=1, size=expect.shape)
    y_obs = expect * np.exp(log_noise * sigma)

    # Select paths and estimate depth
    (
        selected_paths,
        beta_est,
        beta_stderr,
        sigma_est,
        sigma_stderr,
        inv_hessian,
        fit,
        delta_aic,
    ) = sz.deconvolution.estimate_paths(
        X,
        y_obs,
        model=sz.depth_model,
        forward_stop=0.2,
        backward_stop=0.01,
        alpha=alpha,
    )

    assert set(selected_paths) == set(active_paths)
