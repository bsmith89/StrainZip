import numpy as np

import strainzip as sz

# TODO (2024-11-12): Be sure to test the shapes of the beta_stderr results, for instance.


def test_well_specified_deconvolution():
    seed = 0
    n, m = 2, 3  # In-edges / out-edges
    s_samples = 3
    sigma = 1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    np.random.seed(seed)

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_all_paths(n, m)[0]
    assert X.shape == (r_edges, p_paths)

    # Select which pairs of in/out edges are "real" and assign them depths across samples.
    active_paths = sz.deconvolution.simulate_non_redundant_path_indexes(n, m)
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
    model_class, default_model_params = sz.depth_model.NAMED_DEPTH_MODELS["Default"]
    depth_model = model_class(**default_model_params)
    fit = depth_model.fit(y_obs, X_reduced)

    # Calculate likelihood
    assert np.isfinite(fit.get_score("bic"))

    # Estimate standard errors / Check model identifiable.
    assert np.isfinite(fit.stderr_beta).all()


def test_convergence():
    seed = 0
    n, m = 2, 3  # In-edges / out-edges
    s_samples = 3
    sigma = 1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    np.random.seed(seed)

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_all_paths(n, m)[0]
    assert X.shape == (r_edges, p_paths)

    # Select which pairs of in/out edges are "real" and assign them depths across samples.
    active_paths = sz.deconvolution.simulate_non_redundant_path_indexes(n, m)
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

    for model_name in sz.depth_model.VALIDATED_DEPTH_MODELS:
        model_class, default_model_params = sz.depth_model.NAMED_DEPTH_MODELS[
            model_name
        ]
        fit_many_iters = model_class(
            **(default_model_params | dict(maxiter=100000)),
        ).fit(y_obs, X_reduced)
        assert bool(fit_many_iters.converged)

        fit_few_iters = model_class(
            **(default_model_params | dict(maxiter=2)),
        ).fit(y_obs, X_reduced)
        assert not bool(fit_few_iters.converged)


def test_no_noise_offsetlognormal_model_deconvolution():
    model_class, default_model_params = sz.depth_model.NAMED_DEPTH_MODELS[
        "OffsetLogNormal"
    ]
    depth_model = model_class(**default_model_params)
    n, m = 2, 3  # In-edges / out-edges
    s_samples = 3
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_all_paths(n, m)[0]
    assert X.shape == (r_edges, p_paths)

    # Select which pairs of in/out edges are "real" and assign them depths across samples.
    beta = np.array(
        [
            [0e0, 0e0, 0e0],
            [4e0, 4e0, 0e0],
            [1e2, 5e2, 6e3],
            [9e1, 9e1, 2e0],
            [0e0, 0e0, 0e0],
            [0e0, 0e0, 0e0],
        ]
    )

    # Specify the observed depth of each edge.
    expect = X @ (beta * depth_multiplier)
    y_obs = expect

    # Select included paths during the estimation procedure.
    # Possibly over-specified. (see `num_excess_paths`)
    _active_paths = [1, 2, 3]
    X_reduced = X[:, _active_paths]

    # Estimate model parameters
    fit = depth_model.fit(y_obs, X_reduced)
    # Check estimates.
    assert np.allclose(
        fit.beta,
        beta[_active_paths, :],
        atol=1e-3,
        rtol=1e-4,
    )
    assert np.allclose(
        fit.params["sigma"], np.zeros_like(fit.params["sigma"]), atol=1e-4
    )

    # Check BIC
    assert np.allclose(fit.get_score("bic"), 229.4335)

    # Estimate standard errors.
    # Check estimates.
    assert np.allclose(
        fit.stderr_beta,
        np.zeros_like(fit.stderr_beta),
        atol=0.5,
    )


def test_predefined_offsetlognormal_model_deconvolution():
    model_class, default_model_params = sz.depth_model.NAMED_DEPTH_MODELS[
        "OffsetLogNormal"
    ]
    depth_model = model_class(**default_model_params)
    n, m = 2, 3  # In-edges / out-edges
    s_samples = 3
    sigma = 1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_all_paths(n, m)[0]
    assert X.shape == (r_edges, p_paths)

    # Select which pairs of in/out edges are "real" and assign them depths across samples.
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
    fit = depth_model.fit(y_obs, X_reduced)
    # Check estimates.
    assert np.allclose(
        fit.beta,
        np.array(
            [
                [4.5765247e00, 9.9223480e02, -2.7179718e-05],
                [1.9112575e02, -1.0251999e-04, 1.9269342e04],
                [1.0397991e02, -6.5565109e-07, 2.1372384e05],
            ]
        ),
    )
    assert np.allclose(
        fit.params["sigma"],
        np.array([[0.53901196, 0.06199879, 0.37936726]]),
    )

    # Check BIC
    assert np.allclose(fit.get_score("bic"), -63.882275)

    # Estimate standard errors.
    # Check estimates.
    assert np.allclose(
        fit.stderr_beta,
        np.array(
            [
                [3.0417686e00, 4.3542957e01, 3.7935179e-01],
                [7.4494919e01, 6.1989240e-02, 5.1692056e03],
                [4.0011986e01, 4.3839715e-02, 5.7332566e04],
            ]
        ),
    )
