import numpy as np

import strainzip as sz
from strainzip import depth_model


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
    fit = depth_model.fit(y_obs, X_reduced, alpha=alpha)

    # Calculate likelihood
    assert np.isfinite(fit.score)

    # Estimate standard errors / Check model identifiable.
    assert np.isfinite(fit.stderr_beta).all()


def test_no_noise_deconvolution():
    alpha = 1e-5  # Small offset for handling 0s in depths
    n, m = 2, 3  # In-edges / out-edges
    s_samples = 3
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_paths(n, m)[0]
    assert X.shape == (r_edges, p_paths)

    # Select which pairs of in/out edges are "real" and assign them weights across samples.
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
    fit = depth_model.fit(y_obs, X_reduced, alpha=alpha)
    # Check estimates.
    assert np.allclose(
        fit.beta,
        beta[_active_paths, :],
        rtol=1e-4,
    )
    assert np.allclose(fit.sigma, np.zeros_like(fit.sigma), atol=1e-4)

    # Check BIC
    assert np.allclose(fit.score, 208.8452)

    # Estimate standard errors.
    # Check estimates.
    assert np.allclose(
        fit.stderr_beta,
        np.zeros_like(fit.stderr_beta),
        atol=0.5,
    )


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
    fit = depth_model.fit(y_obs, X_reduced, alpha=alpha)
    # Check estimates.
    assert np.allclose(
        fit.beta,
        np.array(
            [
                [4.53333282e00, 9.92237427e02, 3.63797881e-12],
                [1.90782272e02, 1.27329258e-11, 1.92700859e04],
                [1.03953804e02, -6.36646291e-12, 2.13722203e05],
            ]
        ),
    )
    assert np.allclose(
        fit.sigma,
        np.array([[0.5559257], [0.2538453], [0.25384486], [0.05775243], [0.54328245]]),
    )

    # Check BIC
    assert np.allclose(fit.score, -66.81597)

    # Estimate standard errors.
    # Check estimates.
    assert np.allclose(
        fit.stderr_beta,
        np.array(
            [
                [2.6438904e-01, 5.4443096e01, 5.7752453e-07],
                [7.6072998e01, 5.4328339e-06, 7.4474644e03],
                [1.8659252e01, 1.7949537e-06, 3.8362180e04],
            ]
        ),
    )


def test_model_selection_procedure_3x4():
    seed = 1
    alpha = 1e-0  # Small offset for handling 0s in depths
    n, m = 3, 4  # In-edges / out-edges
    s_samples = 10
    sigma = 1e-2  # Scale of the multiplicative noise
    depth_multiplier = 2  # Scaling factor for depths
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
        mean=-1, sigma=4, size=(len(active_paths), s_samples)
    )
    beta = beta.round(1) * depth_multiplier  # Structural zeros

    # Simulate the observed depth of each edge.
    expect = X @ beta
    log_noise = np.random.normal(loc=0, scale=1, size=expect.shape)
    y_obs = expect * np.exp(log_noise * sigma)

    # Select paths and estimate depth
    (selected_paths, delta_score,) = sz.deconvolution.select_paths(
        X,
        y_obs,
        model=depth_model,
        forward_stop=0.0,
        backward_stop=0.0,
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
    (selected_paths, delta_score,) = sz.deconvolution.select_paths(
        X,
        y_obs,
        model=depth_model,
        forward_stop=0.0,
        backward_stop=0.0,
        alpha=alpha,
    )

    assert set(selected_paths) == set(active_paths)
