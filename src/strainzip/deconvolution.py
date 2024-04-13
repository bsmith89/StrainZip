from functools import cache
from itertools import product

import numpy as np
from scipy.stats import chi2


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


def aic_score(loglik, p_paths, e_edges, s_samples):
    n = e_edges * s_samples
    k = p_paths * s_samples
    # TODO?
    # See https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size
    # FIXME: Figure out if this is actually the correct formula.
    # aicc = -2 * loglik + 2 * k + (2 * k**2 + 2*k) / (n - k - 1)
    # return aicc
    return -2 * loglik + 2 * k


def iter_forward_greedy_path_selection(X, y, model, active_paths=None, **kwargs):
    e_edges, p_paths = X.shape
    _e_edges, s_samples = y.shape
    assert e_edges == _e_edges

    all_paths = set(range(p_paths))

    if active_paths is None:
        active_paths = set()
    else:
        active_paths = set(active_paths)
    inactive_paths = all_paths - active_paths

    while inactive_paths:
        scores = []
        for p in inactive_paths:
            trial_paths = active_paths | {p}
            X_trial = X[:, list(trial_paths)]
            beta_est, sigma_est, fit = model.fit(y, X_trial, **kwargs)
            loglik = -model.negloglik(beta_est, sigma_est, y, X_trial, **kwargs)
            aic = aic_score(loglik, len(trial_paths), e_edges, s_samples)
            scores.append((loglik, aic, trial_paths))
        best_loglik, best_aic, best_paths = sorted(scores, reverse=True)[0]
        active_paths = best_paths
        inactive_paths = all_paths - active_paths
        yield list(sorted(active_paths)), best_loglik, {
            tuple(sorted(pp)): a for _, a, pp in scores
        }


def iter_backward_greedy_path_selection(X, y, model, active_paths=None, **kwargs):
    e_edges, p_paths = X.shape
    _e_edges, s_samples = y.shape
    assert e_edges == _e_edges

    all_paths = set(range(p_paths))

    if active_paths is None:
        active_paths = set(all_paths)
    else:
        active_paths = set(active_paths)

    while active_paths:
        scores = []
        for p in active_paths:
            trial_paths = active_paths - {p}
            X_trial = X[:, list(trial_paths)]
            beta_est, sigma_est, fit = model.fit(y, X_trial, **kwargs)
            loglik = -model.negloglik(beta_est, sigma_est, y, X_trial, **kwargs)
            aic = aic_score(loglik, len(trial_paths), e_edges, s_samples)
            scores.append((loglik, aic, trial_paths))
        best_loglik, best_aic, best_paths = sorted(scores, reverse=True)[0]
        active_paths = best_paths
        yield list(sorted(active_paths)), best_loglik, {
            tuple(sorted(pp)): a for _, a, pp in scores
        }


def likelihood_ratio_test(delta_loglik, delta_df):
    test_statistic = 2 * delta_loglik
    p_value = chi2.sf(test_statistic, delta_df)
    return p_value


def estimate_paths(
    X, y, model, forward_stop=0.2, backward_stop=0.01, verbose=0, **kwargs
):
    s_samples = y.shape[1]

    prev_loglik = float("-inf")
    all_aic = {}
    forward_selected_paths = []
    active_paths = []
    for active_paths, loglik, multi_aic in iter_forward_greedy_path_selection(
        X, y, model, **kwargs
    ):
        pvalue = likelihood_ratio_test(
            delta_loglik=loglik - prev_loglik, delta_df=s_samples
        )
        all_aic |= multi_aic
        prev_loglik = loglik
        if verbose >= 2:
            print(active_paths, pvalue)
        if verbose >= 3:
            print(multi_aic)
        if pvalue > forward_stop:
            if verbose >= 1:
                print(
                    f"Stop forward selection with {active_paths} and pvalue: {pvalue}"
                )
            forward_selected_paths = active_paths
            break
    else:
        if verbose >= 1:
            print("All paths added in forward pass without stopping.")
        forward_selected_paths = active_paths

    selected_paths = forward_selected_paths
    reduced_paths = []
    for reduced_paths, loglik, multi_aic in iter_backward_greedy_path_selection(
        X, y, model, active_paths=forward_selected_paths, **kwargs
    ):
        pvalue = likelihood_ratio_test(prev_loglik - loglik, delta_df=s_samples)
        all_aic |= multi_aic
        prev_loglik = loglik
        if verbose >= 2:
            print(reduced_paths, pvalue)
        if verbose >= 3:
            print(multi_aic)
        if pvalue < backward_stop:
            if verbose >= 1:
                print(f"Stop backwards selection with pvalue: {pvalue}")
            break
        else:
            selected_paths = reduced_paths
    else:
        if verbose >= 1:
            print("All paths removed in backward pass without stopping.")
        selected_paths = reduced_paths

    if verbose >= 2:
        print(selected_paths)

    X_selected = X[:, selected_paths]
    beta_est, sigma_est, fit = model.fit(y, X_selected, **kwargs)
    beta_stderr, sigma_stderr, inv_beta_hessian = model.estimate_stderr(
        y, X_selected, beta_est, sigma_est, **kwargs
    )

    # Calculate the delta-AIC relative to the runner-up.
    if verbose >= 4:
        print(all_aic)
    # Make sure that the best model is much better.
    best_aic = all_aic.pop(tuple(selected_paths))
    second_best_aic = min(all_aic.values())
    delta_aic = best_aic - second_best_aic

    return (
        selected_paths,
        beta_est,
        beta_stderr,
        sigma_est,
        sigma_stderr,
        inv_beta_hessian,
        fit,
        delta_aic,
    )


def deconvolve_junction(
    in_vertices,
    in_flows,
    out_vertices,
    out_flows,
    model,
    forward_stop=0.2,
    backward_stop=0.01,
    verbose=False,
    **kwargs,
):
    X, y, labels = formulate_path_deconvolution(in_flows, out_flows)

    (
        selected_paths,
        beta_est,
        beta_stderr,
        sigma_est,
        sigma_stderr,
        inv_beta_hessian,
        fit,
        delta_aic,
    ) = estimate_paths(
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

    return inv_beta_hessian, named_paths, beta_est, delta_aic
