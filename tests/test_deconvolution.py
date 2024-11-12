from collections import Counter
from itertools import chain, product

import numpy as np

import strainzip as sz

# TODO (2024-05-30): BIG refactor of this test set, splitting out tests of the depth model and depth model result interfaces.
# Testing deconvolution means testing the model search procedure and everything in strainzip.deconvolution


def test_minimal_complete_pathset_counts():
    for n, m in product(range(1, 5), repeat=2):
        expected_num = sz.deconvolution.num_minimal_complete_pathsets(n, m)
        all_pathsets = list(sz.deconvolution.iter_all_minimal_complete_pathsets(n, m))
        path_counts = Counter(chain(*all_pathsets))
        assert (
            len(set(path_counts.values())) == 1
        ), "All paths should be seen the same number of times across all pathsets."
        assert len(all_pathsets) == expected_num


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
