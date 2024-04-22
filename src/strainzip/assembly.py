from multiprocessing import Pool

import graph_tool as gt
import graph_tool.search
import graph_tool.topology
import numpy as np

import strainzip as sz
from strainzip import depth_model


def edge_has_no_siblings(g):
    "Check whether upstream or downstream sibling edges exist for every edge."
    v_in_degree = g.degree_property_map("in")
    v_out_degree = g.degree_property_map("out")
    e_num_in_siblings = gt.edge_endpoint_property(g, v_in_degree, "target")
    e_num_out_siblings = gt.edge_endpoint_property(g, v_out_degree, "source")
    e_has_no_sibling_edges = g.new_edge_property(
        "bool", (e_num_in_siblings.a <= 1) & (e_num_out_siblings.a <= 1)
    )
    return e_has_no_sibling_edges


def get_cycles_and_label_maximal_unitigs(g):
    "Assign unitig indices to vertices in maximal unitigs."
    no_sibling_edges = edge_has_no_siblings(g)
    g_filt = gt.GraphView(g, efilt=no_sibling_edges, directed=True)
    cyclic_paths = gt.topology.all_circuits(g_filt)
    # WARNING: Possibly non-deterministic?
    labels, counts = gt.topology.label_components(g_filt, directed=False)
    return cyclic_paths, labels, counts, g_filt


def iter_maximal_unitig_paths(g):
    # FIXME: Fails when any of the unitigs are cycles.
    # Should be able to drop one edge from each cycle
    # in the GraphView and get the correct outputs...
    # Alternatively should be able to run gt.topology.all_circuits
    # and pre-select the circuits to yield as their own sorted lists.
    cyclic_paths, labels, counts, g_filt = get_cycles_and_label_maximal_unitigs(g)
    involved_in_cycle = g_filt.new_vertex_property("bool", val=1)
    for cycle in cyclic_paths:
        involved_in_cycle.a[cycle] = 0
        if len(cycle) < 2:
            continue
        yield cycle

    g_filt_drop_cycles = gt.GraphView(g_filt, vfilt=involved_in_cycle, directed=True)

    sort_order = gt.topology.topological_sort(g_filt_drop_cycles)
    sort_labels = labels.a[sort_order]
    for i, _ in enumerate(counts):
        unitig_path = sort_order[sort_labels == i]
        if len(unitig_path) < 2:
            continue
        yield unitig_path
    # NOTE (2024-04-17): I could sort this output (and make this function
    # a poor excuse for a generator) if I want to be absolutely sure that the
    # unitig order is deterministic.
    # maximal_unitig_paths = []  # <-- aggregate this in the loop above.
    # for path in sorted(maximal_unitig_paths):
    #     return path


def find_tips(g, also_required=None):
    if also_required is None:
        also_required = True
    v_degree = g.degree_property_map("total")
    result = np.where((v_degree.a == 1) & also_required)[
        0
    ]  # np.where returns a tuple (maybe to deal with an N-dimensional mask?).
    return result


def find_junctions(g):
    v_in_degree = g.degree_property_map("in")
    v_out_degree = g.degree_property_map("out")
    result = np.where(
        ((v_in_degree.a > 1) & (v_out_degree.a >= 1))
        | ((v_in_degree.a >= 1) & (v_out_degree.a > 1))
    )[
        0
    ]  # np.where returns a tuple (maybe to deal with an N-dimensional mask?).
    # FIXME (2024-04-19): Does the above deal with vertex filtering correctly??
    # My mental model would be that it would include a bunch of filtered-out
    # vertices in the list. I guess there's a chance that all of these have
    # in/out degree of 0...?
    return result


def backlinked_graph(graph):
    out = graph.copy()
    edges = graph.get_edges()
    rev_edges = edges[:, [1, 0]]
    out.add_edge_list(rev_edges)
    return out


def get_shortest_distance(graph, root, weights, max_length=None):
    original_graph = graph
    graph = backlinked_graph(graph)
    weights = graph.own_property(weights)
    edge_weights = gt.edge_endpoint_property(graph, weights, "source")
    dist = gt.topology.shortest_distance(
        graph, root, weights=edge_weights, directed=True, max_dist=max_length
    )
    return original_graph.own_property(dist)


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
    fit, paths, named_paths, score_margin = sz.deconvolution.deconvolve_junction(
        in_neighbors,
        in_flows,
        out_neighbors,
        out_flows,
        model=depth_model,  # TODO (2024-04-20): Allow this to be passed in by changing it from a module into a class.
        forward_stop=forward_stop,
        backward_stop=backward_stop,
        alpha=alpha,
    )

    X = sz.deconvolution.design_paths(n, m)[0]

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
                    sz.assembly.find_junctions(graph), graph, flow, max_paths=max_paths
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
