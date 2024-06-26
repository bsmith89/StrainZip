from collections import Counter
from itertools import chain, product

import numpy as np

import strainzip as sz
import strainzip.depth_model

# TODO (2024-05-30): BIG refactor of this test set, splitting out tests of the depth model and depth model result interfaces.
# Testing deconvolution means testing the model search procedure and everything in strainzip.deconvolution
# Testing depth models is just a part of this (and should be isolated)
# Be sure to test the shapes of the beta_stderr results, for instance.


def test_minimal_complete_pathset_counts():
    for n, m in product(range(1, 5), repeat=2):
        expected_num = sz.deconvolution.num_minimal_complete_pathsets(n, m)
        all_pathsets = list(sz.deconvolution.iter_all_minimal_complete_pathsets(n, m))
        path_counts = Counter(chain(*all_pathsets))
        assert (
            len(set(path_counts.values())) == 1
        ), "All paths should be seen the same number of times across all pathsets."
        assert len(all_pathsets) == expected_num


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

    for model_name in sz.depth_model.NAMED_DEPTH_MODELS:
        model_class, default_model_params = sz.depth_model.NAMED_DEPTH_MODELS[
            model_name
        ]
        fit_many_iters = model_class(
            maxiter=100000,
            **default_model_params,
        ).fit(y_obs, X_reduced)
        assert fit_many_iters.converged

        fit_few_iters = model_class(
            maxiter=2,
            **default_model_params,
        ).fit(y_obs, X_reduced)
        assert not fit_few_iters.converged


def test_no_noise_deconvolution():
    model_class, default_model_params = sz.depth_model.NAMED_DEPTH_MODELS["Default"]
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
        rtol=1e-4,
    )
    assert np.allclose(fit.sigma, np.zeros_like(fit.sigma), atol=1e-4)

    # Check BIC
    assert np.allclose(fit.get_score("bic"), 203.59065)

    # Estimate standard errors.
    # Check estimates.
    assert np.allclose(
        fit.stderr_beta,
        np.zeros_like(fit.stderr_beta),
        atol=0.5,
    )


def test_predefined_deconvolution():
    model_class, default_model_params = sz.depth_model.NAMED_DEPTH_MODELS["Default"]
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
                [4.53333282e00, 9.92237427e02, 3.63797881e-12],
                [1.90782272e02, 1.27329258e-11, 1.92700859e04],
                [1.03953804e02, -6.36646291e-12, 2.13722203e05],
            ]
        ),
    )
    assert np.allclose(
        fit.sigma,
        np.array([[0.5422827, 0.06206119, 0.3793803]]),
    )

    # Check BIC
    assert np.allclose(fit.get_score("bic"), -63.953182)

    # Estimate standard errors.
    # Check estimates.
    assert np.allclose(
        fit.stderr_beta,
        np.array(
            [
                [2.4824376e00, 4.3543179e01, 3.7938039e-06],
                [7.4405067e01, 6.2061298e-07, 5.1694292e03],
                [3.9861290e01, 4.3883844e-07, 5.7333625e04],
            ]
        ),
    )


# FIXME: Parameterize the previous test instead of making this nearly identical test.
def test_model_selection_procedure_2x1():
    seed = 0
    model_class, default_model_params = sz.depth_model.NAMED_DEPTH_MODELS["Default"]
    depth_model = model_class(**default_model_params)
    n, m = 2, 1  # In-edges / out-edges
    s_samples = 4
    sigma = 1e-1  # Scale of the multiplicative noise
    depth_multiplier = 1  # Scaling factor for depths
    num_excess_paths = 0  # How many extra paths to include beyond correct ones.

    np.random.seed(seed)

    r_edges, p_paths = (n + m, n * m)
    X = sz.deconvolution.design_all_paths(n, m)[0]
    assert X.shape == (r_edges, p_paths)

    # Select which pairs of in/out edges are "real" and assign them depths across samples.
    active_paths = sz.deconvolution.simulate_non_redundant_path_indexes(
        n, m, excess=num_excess_paths
    )
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
    # NOTE: The below is a hack to shoe-horn the new deconvolution module
    # into an old test.
    scores = sz.deconvolution.greedy_search_potential_pathsets(
        y_obs[:n],
        y_obs[-m:],
        model=depth_model,
        score_name="bic",
    )
    top_scores = list(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    pathset, best_score = top_scores[0]
    _, second_score = top_scores[1]
    score_margin = best_score - second_score
    selected_paths = [
        sz.deconvolution.raveled_coords(p.left, p.right, n, m) for p in pathset
    ]

    assert set(selected_paths) == set(active_paths)
