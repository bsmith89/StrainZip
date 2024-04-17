from functools import cache
from itertools import product

import numpy as np


@cache
def design_paths(n, m):
    in_designs = np.eye(n, dtype=int)
    out_designs = np.eye(m, dtype=int)
    design_products = product(in_designs, out_designs)
    label_products = product(range(n), range(m))
    design = np.stack(
        [np.concatenate([in_row, out_row]) for in_row, out_row in design_products],
        axis=1,
    )
    return design, list(label_products)


def simulate_active_paths(n, m, excess=0):
    def raveled_coords(i, j, n, m):
        assert i < n
        assert j < m
        return i * m + j

    a = min(n, m)
    b = max(n, m) - a

    # Pick initial pairs by picking a subset of columns in a permutation matrix
    perm_mat = np.eye(a)[:, np.random.choice(a, replace=False, size=a)]
    # Pick additional pairs by adding additional columns from a second permutation matrix.
    edge = np.eye(a)[:, np.random.choice(a, replace=False, size=b)]
    mapping = np.concatenate([perm_mat, edge], axis=1)
    mapping = list(enumerate(np.argmax(mapping, axis=0)))

    # Add additional mappings beyond minimal.
    excess_pairs = [
        (i, j) for i, j in product(range(a + b), range(a)) if (i, j) not in mapping
    ]
    mapping += [
        excess_pairs[i]
        for i in np.random.choice(len(excess_pairs), replace=False, size=excess)
    ]

    if n < m:
        active_paths = [(raveled_coords(i, j, n, m), (i, j + n)) for j, i in mapping]
    else:
        active_paths = [(raveled_coords(i, j, n, m), (i, j + n)) for i, j in mapping]

    return sorted(active_paths)


def formulate_path_deconvolution(in_flows, out_flows):
    n, m = in_flows.shape[0], out_flows.shape[0]
    s = in_flows.shape[1]
    assert in_flows.shape[1] == out_flows.shape[1]
    X, labels = design_paths(n, m)
    y = np.concatenate([in_flows, out_flows], axis=0)
    return X, y, labels


# def aic_score(loglik, p_paths, e_edges, s_samples):
#     n = e_edges * s_samples
#     k = p_paths * s_samples
#     # TODO?
#     # See https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size
#     # FIXME: Figure out if this is actually the correct formula.
#     # aicc = -2 * loglik + 2 * k + (2 * k**2 + 2*k) / (n - k - 1)
#     # return aicc
#     return -2 * loglik + 2 * k


def iter_forward_greedy_path_selection(X, y, model, init_paths=None, **kwargs):
    p_paths = X.shape[1]
    all_paths = set(range(p_paths))

    if init_paths is None:
        active_paths = set()
    else:
        active_paths = set(init_paths)
    inactive_paths = all_paths - active_paths

    while inactive_paths:
        scores = []
        for p in inactive_paths:
            trial_paths = active_paths | {p}
            X_trial = X[:, list(trial_paths)]
            fit = model.fit(y, X_trial, **kwargs)
            scores.append((fit.score, trial_paths))
        _, best_paths = sorted(scores, reverse=True)[0]
        active_paths = best_paths
        inactive_paths = all_paths - active_paths
        yield tuple(sorted(active_paths)), {tuple(sorted(pp)): s for s, pp in scores}


def iter_backward_greedy_path_selection(X, y, model, init_paths=None, **kwargs):
    p_paths = X.shape[1]
    all_paths = set(range(p_paths))

    if init_paths is None:
        active_paths = set(all_paths)
    else:
        active_paths = set(init_paths)

    while active_paths:
        scores = []
        for p in active_paths:
            trial_paths = active_paths - {p}
            X_trial = X[:, list(trial_paths)]
            fit = model.fit(y, X_trial, **kwargs)
            scores.append((fit.score, trial_paths))
        _, best_paths = sorted(scores, reverse=True)[0]
        active_paths = best_paths
        yield tuple(sorted(active_paths)), {tuple(sorted(pp)): s for s, pp in scores}


def select_paths(X, y, model, forward_stop, backward_stop, verbose=False, **kwargs):
    curr_score = np.nan
    all_scores = {}
    active_paths = ()
    for active_paths, multi_scores in iter_forward_greedy_path_selection(
        X, y, model, init_paths=[], **kwargs
    ):
        all_scores |= multi_scores
        prev_score = curr_score
        curr_score = all_scores[active_paths]
        if verbose:
            print(active_paths, curr_score)
        delta_score = curr_score - prev_score
        if delta_score < forward_stop:
            break

    prev_active_paths = active_paths
    for active_paths, multi_scores in iter_backward_greedy_path_selection(
        X, y, model, init_paths=active_paths, **kwargs
    ):
        all_scores |= multi_scores
        prev_score = curr_score
        curr_score = all_scores[active_paths]
        if verbose:
            print(active_paths, curr_score)
        delta_score = curr_score - prev_score
        if delta_score < backward_stop:
            active_paths = prev_active_paths  # Backtrack
            break
        else:
            prev_active_paths = active_paths

    # How does this compare to the best model seen?
    curr_score = all_scores.pop(active_paths)
    compare_score = max(all_scores.values())
    delta_score = curr_score - compare_score

    return (
        active_paths,
        delta_score,
    )


def deconvolve_junction(
    in_vertices,
    in_flows,
    out_vertices,
    out_flows,
    model,
    forward_stop=0,
    backward_stop=0,
    **kwargs,
):
    X, y, labels = formulate_path_deconvolution(in_flows, out_flows)

    selected_paths, delta_aic = select_paths(
        X,
        y,
        model=model,
        forward_stop=forward_stop,
        backward_stop=backward_stop,
        **kwargs,
    )
    named_paths = []
    for path_idx in selected_paths:
        left = in_vertices[labels[path_idx][0]]
        right = out_vertices[labels[path_idx][1]]
        named_paths.append((left, right))

    fit = model.fit(y, X[:, selected_paths], **kwargs)

    return fit, selected_paths, named_paths, delta_aic
