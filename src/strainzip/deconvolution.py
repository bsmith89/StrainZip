from functools import cache
from itertools import product
from multiprocessing import Pool as processPool
from multiprocessing.dummy import Pool as threadPool

import numpy as np

from . import depth_model
from .assembly import find_junctions


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
    verbose=False,
    **kwargs,
):
    X, y, labels = formulate_path_deconvolution(in_flows, out_flows)

    selected_paths, delta_aic = select_paths(
        X,
        y,
        model=model,
        forward_stop=forward_stop,
        backward_stop=backward_stop,
        verbose=verbose,
        **kwargs,
    )
    named_paths = []
    for path_idx in selected_paths:
        left = in_vertices[labels[path_idx][0]]
        right = out_vertices[labels[path_idx][1]]
        named_paths.append((left, right))

    fit = model.fit(y, X[:, selected_paths], **kwargs)

    return fit, selected_paths, named_paths, delta_aic


def _iter_junction_deconvolution_data(junction_iter, graph, flow, max_paths):
    for j in junction_iter:
        in_neighbors = graph.get_in_neighbors(j)
        out_neighbors = graph.get_out_neighbors(j)
        n, m = len(in_neighbors), len(out_neighbors)
        if n * m > max_paths:
            continue

        # Collect flows
        # print(in_neighbors, j, out_neighbors)
        in_flows = np.stack([flow[(i, j)] for i in in_neighbors])
        out_flows = np.stack([flow[(j, i)] for i in out_neighbors])

        # FIXME (2024-04-20): Decide if I actually want to
        # balance flows before fitting.
        log_offset_ratio = np.log(in_flows.sum()) - np.log(out_flows.sum())
        in_flows = np.exp(np.log(in_flows) - log_offset_ratio / 2)
        out_flows = np.exp(np.log(out_flows) + log_offset_ratio / 2)
        yield j, in_neighbors, in_flows, out_neighbors, out_flows


# TODO (2024-04-20): Consider moving these deconvolution functions into the assembly app
# instead of the assembly module (which was really supposed to be *topology* not assembly.
def _calculate_junction_deconvolution(args):
    (
        junction,
        in_neighbors,
        in_flows,
        out_neighbors,
        out_flows,
        forward_stop,
        backward_stop,
        alpha,
        score_margin_thresh,
        condition_thresh,
    ) = args
    n, m = len(in_neighbors), len(out_neighbors)
    fit, paths, named_paths, score_margin = deconvolve_junction(
        in_neighbors,
        in_flows,
        out_neighbors,
        out_flows,
        model=depth_model,  # TODO (2024-04-20): Allow this to be passed in by changing it from a module into a class.
        forward_stop=forward_stop,
        backward_stop=backward_stop,
        alpha=alpha,
    )

    X = design_paths(n, m)[0]

    if not (score_margin > score_margin_thresh):
        # print(f"[junc={j} / {n}x{m}] Cannot pick best model. (Selected model had {len(paths)} paths; score margin: {score_margin})")
        pass
    elif not X[:, paths].sum(1).min() == 1:
        # print(f"[junc={j} / {n}x{m}] Non-complete. (Best model had {len(paths)} paths; score margin: {score_margin})")
        pass
    elif not len(paths) <= max(n, m):
        # print(f"[junc={j} / {n}x{m}] Non-minimal. (Best model had {len(paths)} paths; score margin: {score_margin})")
        pass
    elif not (np.linalg.cond(fit.hessian_beta) < condition_thresh):
        # print(f"[junc={j} / {n}x{m}] Non-identifiable. (Best model had {len(paths)} paths; score margin: {score_margin})")
        pass
    else:
        # print(f"[junc={j} / {n}x{m}] SUCCESS! Selected {len(paths)} paths; score margin: {score_margin}")
        return junction, named_paths, {"path_depths": np.array(fit.beta.clip(0))}


def parallel_calculate_all_junction_deconvolutions(
    graph,
    flow,
    forward_stop=0.0,
    backward_stop=0.0,
    alpha=1.0,
    score_margin_thresh=20.0,
    condition_thresh=1e5,
    max_paths=20,
    processes=1,
):
    # FIXME (2024-04-21): This architecture means that all of the JAX
    # stuff needs to be recompiled every time in every process.
    # If I'm lucky persistent compilation cache will solve my problems
    # some day: https://github.com/google/jax/discussions/13736
    if processes > 1:
        Pool = processPool
    else:
        Pool = threadPool

    with Pool(processes=processes) as pool:
        deconv_results = pool.imap_unordered(
            _calculate_junction_deconvolution,
            (
                (
                    junction,
                    in_neighbors,
                    in_flows,
                    out_neighbors,
                    out_flows,
                    forward_stop,
                    backward_stop,
                    alpha,
                    score_margin_thresh,
                    condition_thresh,
                )
                for junction, in_neighbors, in_flows, out_neighbors, out_flows in _iter_junction_deconvolution_data(
                    find_junctions(graph), graph, flow, max_paths=max_paths
                )
            ),
        )

        batch = []
        for result in deconv_results:
            if result is not None:
                junction, named_paths, path_depths_dict = result
                # print(f"{junction}: {named_paths}", end=" | ")
                batch.append((junction, named_paths, path_depths_dict))

        return batch
