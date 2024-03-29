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
    delta_aic, paths, weights = sz.deconvolution.fit_sparse_paths_exhaustive(
        design, observed
    )
    assert np.allclose(delta_aic, -2.0)
    assert paths == (0, 1)
    assert [labels[p] for p in paths] == [(0, 0), (0, 1)]


def test_deconvolution_problem_estimation():
    delta_aic, named_paths, depths = sz.deconvolution.estimate_path_weights(
        [2, 1],
        np.array([[100, 0], [20, 0], [50, 0]]),
        [5, 6, 7],
        np.array([[100, 0, 0], [10, 10.0, 0], [0, 50.0, 0.0]]),
    )
    assert named_paths == [(2, 5), (2, 6)]
