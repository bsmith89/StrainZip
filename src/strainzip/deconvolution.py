from dataclasses import dataclass
from functools import cache
from itertools import chain, combinations, permutations, product
from typing import Any, FrozenSet

import numpy as np
import scipy as sp


@cache
def design_all_paths(n, m):
    in_designs = np.eye(n, dtype=int)
    out_designs = np.eye(m, dtype=int)
    design_products = product(in_designs, out_designs)
    label_products = product(range(n), range(m))
    design = np.stack(
        [np.concatenate([in_row, out_row]) for in_row, out_row in design_products],
        axis=1,
    )
    return design, list(label_products)


def raveled_coords(i, j, n, m):
    assert i < n
    assert j < m
    return i * m + j


def simulate_non_redundant_path_indexes(n, m, excess=0):
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


@dataclass
class LocalPath:
    left: Any  # Integer
    right: Any  # Integer

    def __hash__(self):
        return hash((self.left, self.right))

    def __gt__(self, other):
        return (self.left, self.right) > (other.left, other.right)


def swap_edges(pathA: LocalPath, pathB: LocalPath):
    return LocalPath(pathA.left, pathB.right), LocalPath(pathB.left, pathA.right)


@dataclass
class PathSet:
    paths: FrozenSet[LocalPath]
    n: int
    m: int

    def iter_drops(self):
        for to_drop in self.paths:
            yield PathSet(self.paths - {to_drop}, self.n, self.m)

    def iter_adds(self):
        for left, right in product(range(self.n), range(self.m)):
            to_add = LocalPath(left, right)
            if to_add not in self.paths:
                yield PathSet(self.paths | {to_add}, self.n, self.m)

    def iter_swaps(self):
        for pathA, pathB in combinations(self.paths, 2):
            if (pathA.left == pathB.left) | (pathA.right == pathB.right):
                continue
            else:
                paths = (self.paths - {pathA, pathB}) | set(swap_edges(pathA, pathB))
                yield PathSet(paths, self.n, self.m)

    def iter_neighbors(self):
        return chain(self.iter_drops(), self.iter_swaps(), self.iter_adds())

    def __iter__(self):
        return iter(sorted(self.paths))

    def __hash__(self):
        return hash(self.paths)

    def __len__(self):
        return len(self.paths)

    @property
    def empty(self):
        return len(self.paths) == 0


def num_minimal_complete_pathsets(n, m):
    """Calculate number of pathsets that exist that are minimal and complete.

    To derive, consider an NxM node where N > M:

    We'll line up our larger set {1...N} (the order is arbitrary but fixed for all pathes)
    and match up the elements of {1...M} elements to the top M of this list.
    Then we'll choose  an ordering with replacement of N - M elements out of {1...M}
    to be redundantly matched to the remainder {M...N}.
    """
    n = np.array(n, dtype=float)
    m = np.array(m, dtype=float)
    n, m = np.maximum(n, m), np.minimum(n, m)
    return sp.special.perm(m, m, exact=False) * m ** (n - m)


def iter_all_minimal_complete_pathsets(n, m):
    # Assuming that M < N
    # Pick one, arbitrary order of the longer list (presumably index order [1...N])
    # All pathsets can be constructed by then taking the shorter list and:
    # Take every permutation of the shorter list (M! of these)
    # Then append every random sample of the shorter list of size |N-M|, with replacement. (M^(N-M) of these)
    # Match this list of length max(N, M) with the longer list to construct all pairs.

    flipped = n < m
    if flipped:
        n, m = m, n
    fixed_order = range(n)
    for permutation_of_m in permutations(range(m), r=m):
        for remaining_sample in product(range(m), repeat=(n - m)):
            selected_order = list(permutation_of_m) + list(remaining_sample)
            if not flipped:
                yield PathSet(
                    frozenset(
                        LocalPath(*pair) for pair in zip(fixed_order, selected_order)
                    ),
                    n=n,
                    m=m,
                )
            else:
                yield PathSet(
                    frozenset(
                        LocalPath(*pair) for pair in zip(selected_order, fixed_order)
                    ),
                    n=m,
                    m=n,
                )


@cache
def path_to_design_col(path: LocalPath, n: int, m: int):
    if path.left >= n:
        raise ValueError(f"Left edge {path.left} invalid: >= {n}.")
    if path.right >= m:
        raise ValueError(f"Right edge {path.right} invalid: >= {m}.")
    in_designs = np.eye(n, dtype=int)
    out_designs = np.eye(m, dtype=int)
    return np.concatenate([in_designs[path.left], out_designs[path.right]])


def pathset_to_design(paths: PathSet):
    n, m = paths.n, paths.m
    cols = [path_to_design_col(p, n, m) for p in sorted(paths)]
    if len(cols) == 0:
        return np.ones((n + m, 0))
    else:
        return np.stack(cols, axis=1)


def greedy_search_potential_pathsets(
    in_flows,
    out_flows,
    model,
    score_name="bic",
    verbose=False,
):
    n, m = in_flows.shape[0], out_flows.shape[0]
    y = np.concatenate([in_flows, out_flows])
    # TODO (2024-05-17): Figure out if NaN log-likelihoods are driving the failure to pick any paths.
    # NOTE: Trying to be smarter about picking first path?
    top_inflow, top_outflow = np.argmax(in_flows.sum(1)), np.argmax(out_flows.sum(1))
    curr_pathset = PathSet(
        frozenset([LocalPath(top_inflow, top_outflow)]), n, m
    )  # Best contender for a single path.
    # curr_pathset = PathSet(frozenset(), n, m)  # Empty
    curr_score = model.fit(y, pathset_to_design(curr_pathset)).get_score(score_name)
    curr_score = np.nan_to_num(curr_score, nan=-np.inf)
    scores = {curr_pathset: curr_score}
    while True:
        if verbose:
            print(f"{curr_score}: {curr_pathset}")
        for next_pathset in curr_pathset.iter_neighbors():
            if next_pathset in scores:
                continue
            else:
                next_score = model.fit(y, pathset_to_design(next_pathset)).get_score(
                    score_name
                )
                scores[next_pathset] = np.nan_to_num(next_score, nan=-np.inf)

        top_scores = list(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        best_pathset, best_score = top_scores[0]
        if best_pathset == curr_pathset:
            curr_pathset = best_pathset
            curr_score = best_score
            break
        elif best_score > curr_score:
            curr_pathset = best_pathset
            curr_score = best_score

    # TODO: Return the best pathset itself, instead of the score list. Consider returning the fit, as well.
    return scores


def exhaustive_fit_minimal_complete_pathsets(
    in_flows,
    out_flows,
    model,
    include_empty_pathset=False,
    score_name="bic",
    verbose=False,
):
    n, m = in_flows.shape[0], out_flows.shape[0]
    y = np.concatenate([in_flows, out_flows])

    if include_empty_pathset:
        # Add the most-reduced (noise only) model.
        empty_pathset = PathSet(
            frozenset([]), n, m
        )  # Best contender for a single path.
        empty_score = model.fit(y, pathset_to_design(empty_pathset)).get_score(
            score_name
        )
        empty_score = np.nan_to_num(empty_score, nan=-np.inf)
        scores = {empty_pathset: empty_score}
    else:
        scores = {}

    for pathset in iter_all_minimal_complete_pathsets(n, m):
        s = model.fit(y, pathset_to_design(pathset)).get_score(score_name)
        scores[pathset] = np.nan_to_num(s, nan=-np.inf)
    if verbose:
        print(scores)
    return scores


def deconvolve_junction_exhaustive(
    in_vertices,
    in_flows,
    out_vertices,
    out_flows,
    model,
    score_name="bic",
    verbose=False,
):
    n, m = len(in_vertices), len(out_vertices)

    scores = exhaustive_fit_minimal_complete_pathsets(
        in_flows,
        out_flows,
        model,
        include_empty_pathset=False,
        score_name=score_name,
        verbose=False,
    )

    top_scores = list(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    if verbose:
        for _paths, _score in top_scores:
            print(_score, _paths)

    pathset, best_score = top_scores[0]

    # Check the second best score.
    if len(top_scores) > 1:
        _, second_score = top_scores[1]
    else:
        second_score = -np.inf

    # Calculate how much better the best score is.
    if not np.isfinite(best_score):
        # TODO (2024-06-06): Figure out why this would ever happen...
        score_margin = -np.inf
    else:
        score_margin = best_score - second_score

    y = np.concatenate([in_flows, out_flows])
    X = pathset_to_design(pathset)
    fit = model.fit(y, X)
    named_paths = [(in_vertices[p.left], out_vertices[p.right]) for p in pathset]
    selected_paths = [raveled_coords(p.left, p.right, n, m) for p in pathset]
    # NOTE (2024-05-21): Iterating through a PathSet is now in sorted order,
    # however, it's not obvious that this will always solve the issue of
    # named_paths and fit.beta being in the same order...

    return fit, selected_paths, named_paths, score_margin


def deconvolve_junction_with_search(
    in_vertices,
    in_flows,
    out_vertices,
    out_flows,
    model,
    score_name="bic",
    verbose=False,
):
    n, m = len(in_vertices), len(out_vertices)

    scores = greedy_search_potential_pathsets(
        in_flows, out_flows, model, score_name=score_name, verbose=False
    )

    top_scores = list(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    if verbose:
        for _paths, _score in top_scores:
            print(_score, _paths)

    pathset, best_score = top_scores[0]
    _, second_score = top_scores[1]
    if not np.isfinite(best_score):
        score_margin = -np.inf
    else:
        score_margin = best_score - second_score

    y = np.concatenate([in_flows, out_flows])
    X = pathset_to_design(pathset)
    fit = model.fit(y, X)
    named_paths = [(in_vertices[p.left], out_vertices[p.right]) for p in pathset]
    selected_paths = [raveled_coords(p.left, p.right, n, m) for p in pathset]
    # NOTE (2024-05-21): Iterating through a PathSet is now in sorted order,
    # however, it's not obvious that this will always solve the issue of
    # named_paths and fit.beta being in the same order...

    return fit, selected_paths, named_paths, score_margin
