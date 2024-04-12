import numpy as np

import strainzip as sz


def test_deconvolution_problem_formulation():
    design, observed, labels = sz.deconvolution.formulate_path_decomposition(
        np.array([[100, 0], [20, 0], [50, 0]]),
        np.array([[100, 0, 0], [10, 10.0, 0], [0, 50.0, 0.0]]),
    )
    assert np.array_equal(
        design,
        [
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1],
        ],
    )
    assert np.array_equal(
        observed,
        [
            [100.0, 0.0, 100.0, 0.0, 0.0],
            [20.0, 0.0, 10.0, 10.0, 0.0],
            [50.0, 0.0, 0.0, 50.0, 0.0],
        ],
    )


def test_well_specified_deconvolution():
    model = sz.model_zoo.multiplicative_gaussian_noise
    seed = 0
    alpha = 1e-5  # Small offset for handling 0s in depths
    n, m = 2, 3  # In-edges / out-edges
    s_samples = 3
    sigma = 1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    np.random.seed(seed)

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_paths(n, m)[0].T
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
    beta_est, sigma_est, _ = model.fit(y_obs, X_reduced, alpha=alpha)

    # Calculate likelihood
    loglik = -model.negloglik(beta_est, sigma_est, y_obs, X_reduced, alpha=alpha)
    assert np.isfinite(loglik)

    # Estimate standard errors.
    beta_stderr, sigma_stderr, inv_beta_hessian = model.estimate_stderr(
        y_obs,
        X_reduced,
        beta_est,
        sigma_est,
        alpha=alpha,
    )

    # Check model identifiable.
    assert np.isfinite(beta_stderr).all()
    assert np.isfinite(sigma_stderr)


def test_predefined_deconvolution():
    model = sz.model_zoo.multiplicative_gaussian_noise
    seed = 0
    alpha = 1e-5  # Small offset for handling 0s in depths
    n, m = 2, 3  # In-edges / out-edges
    s_samples = 3
    sigma = 1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_paths(n, m)[0].T
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
    beta_reduced_est, sigma_est, _ = model.fit(y_obs, X_reduced, alpha=alpha)
    # Check estimates.
    assert np.allclose(
        beta_reduced_est,
        np.array(
            [
                [4.5331750e00, 9.9224597e02, -3.8744474e-10],
                [1.9078610e02, -3.6834535e-10, 1.9270270e04],
                [1.0395310e02, 1.4642865e-10, 2.1371894e05],
            ]
        ),
    )
    assert np.allclose(sigma_est, np.array([0.3837776]))

    # Check likelihood
    loglik = -model.negloglik(
        beta_reduced_est, sigma_est, y_obs, X_reduced, alpha=alpha
    )
    assert np.allclose(loglik, -6.918624)

    # Estimate standard errors.
    beta_reduced_stderr, sigma_stderr, inv_beta_hessian = model.estimate_stderr(
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
                [1.7567477e00, 2.6926828e02, 3.8375520e-06],
                [5.2658543e01, 3.8375629e-06, 5.2294214e03],
                [2.8209854e01, 2.7137764e-06, 5.7996836e04],
            ]
        ),
    )
    assert np.allclose(sigma_stderr, np.array([[0.07006839]]))


def test_model_selection_procedure():
    model = sz.model_zoo.multiplicative_gaussian_noise
    seed = 0
    alpha = 1e-0  # Small offset for handling 0s in depths
    n, m = 3, 4  # In-edges / out-edges
    s_samples = 4
    sigma = 1e-1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 1  # How many extra paths to include beyond correct ones.

    np.random.seed(seed)

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_paths(n, m)[0].T
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
    ) = sz.deconvolution.estimate_paths(
        X,
        y_obs,
        model=sz.model_zoo.multiplicative_gaussian_noise,
        forward_stop=0.2,
        backward_stop=0.01,
        alpha=alpha,
    )

    assert set(selected_paths) == set(active_paths)
