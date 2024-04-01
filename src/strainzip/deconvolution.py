from functools import cache
from itertools import combinations, product
from typing import Any

import numpy as np
from sklearn.decomposition import non_negative_factorization


@cache
def design_paths(n, m):
    in_designs = np.eye(n, dtype=int)
    out_designs = np.eye(m, dtype=int)
    design_products = product(in_designs, out_designs)
    label_products = product(range(n), range(m))
    design = np.stack(
        [np.concatenate([in_row, out_row]) for in_row, out_row in design_products]
    )
    return design, list(label_products)


def formulate_path_decomposition(in_flows, out_flows):
    n, m = in_flows.shape[1], out_flows.shape[1]
    assert in_flows.shape[0] == out_flows.shape[0]
    design, labels = design_paths(n, m)
    observed = np.concatenate([in_flows, out_flows], axis=1)
    return design, observed, labels


def residual_flow(path_weights, design, observed):
    expect = path_weights @ design
    resid = observed - expect
    return resid


def aic_score(resid, k):
    rss = np.sum(resid**2)
    n = resid.shape[0] + resid.shape[1]
    df = n - k
    assert rss > 0
    return n * np.log(rss / n) + 2 * k


def adhoc_score(resid, k, penalty=2):
    absolute_sum_of_residuals = np.sum(np.abs(resid))
    return absolute_sum_of_residuals + k * penalty


def optimize_path_weights(
    design, observed, solver="cd", max_iter=20_000, random_state=0, **kwargs
):
    k = design.shape[0]
    W, _, _ = non_negative_factorization(
        observed,
        H=design.astype(float),
        update_H=False,
        n_components=k,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs,
    )
    return W


def fit_paths_exhaustive(
    full_design,
    observed,
    fit_func=optimize_path_weights,
    score_func=aic_score,
    return_all_results=False,
    fit_kwargs=None,
    score_kwargs=None,
):
    if fit_kwargs is None:
        fit_kwargs = {}
    if score_kwargs is None:
        score_kwargs = {}

    full_k = full_design.shape[0]
    results = {}
    for k in range(1, full_k + 1):
        for paths in combinations(range(full_k), k):
            _paths = list(paths)
            reduced_design = np.zeros_like(full_design)
            reduced_design[_paths] = full_design[_paths]
            weights = fit_func(reduced_design, observed, **fit_kwargs)
            resid = residual_flow(weights, full_design, observed)
            score = score_func(resid, len(paths), **score_kwargs)
            # import pdb; pdb.set_trace()
            results[paths] = (score, weights)
    scores, ranked = zip(*sorted([(results[k][0], k) for k in results]))
    if return_all_results:
        return results
    return scores[0] - scores[1], ranked[0], results[ranked[0]][1]


def estimate_path_weights(in_vertices, in_flows, out_vertices, out_flows, **kwargs):
    n = len(in_vertices)
    m = len(out_vertices)
    design, observed, labels = formulate_path_decomposition(in_flows, out_flows)
    delta_aic, paths, weights = fit_paths_exhaustive(design, observed, **kwargs)

    named_paths = []
    depths = []
    for path_idx in paths:
        left = in_vertices[labels[path_idx][0]]
        right = out_vertices[labels[path_idx][1]]
        named_paths.append((left, right))
        depths.append(weights[:, path_idx])

    return delta_aic, named_paths, depths
