import logging
from typing import List

import graph_tool as gt
import graph_tool.topology
import numpy as np

from .logging_util import phase_debug, tqdm_debug


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
    # np.where returns a tuple (maybe to deal with an N-dimensional mask?).
    result = np.where((v_degree.a == 1) & also_required)[0]
    return result


def find_junctions(g, also_required=None):
    if also_required is None:
        also_required = True

    v_in_degree = g.degree_property_map("in")
    v_out_degree = g.degree_property_map("out")
    is_junction = ((v_in_degree.a > 1) & (v_out_degree.a >= 1)) | (
        (v_in_degree.a >= 1) & (v_out_degree.a > 1)
    )
    # np.where returns a tuple (maybe to deal with an N-dimensional mask?).
    result = np.where(is_junction & also_required)[0]
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


def get_shortest_distance(graph, roots: List[int], weights, max_length=None):
    # TODO: Write a test for this function:
    #   - Make sure that a single root is at distance 0 from itself.
    #   - Make sure that filtering doesn't break anything?
    #   - Check that _both_ in-neighbors and out-neighbors of the root are at a distance
    #     equal to the length of the root vertex.
    original_graph = graph
    graph = backlinked_graph(graph)
    weights = graph.own_property(weights)
    edge_weights = gt.edge_endpoint_property(graph, weights, "source")

    min_dist = np.inf * np.ones(graph.num_vertices(ignore_filter=True))
    for v in roots:
        dist = gt.topology.shortest_distance(
            graph, v, weights=edge_weights, directed=True, max_dist=max_length
        )
        min_dist[dist.a < min_dist] = dist.a[dist.a < min_dist]

    return original_graph.new_vertex_property("float", vals=min_dist)


def vertex_or_neighbors(graph, vprop):
    "Boolean vertex property: True if vertex or any neighbors are True."
    left_true = gt.edge_endpoint_property(graph, prop=vprop, endpoint="source")
    right_true = gt.edge_endpoint_property(graph, prop=vprop, endpoint="target")
    left_neighbor_true = gt.incident_edges_op(graph, "in", "max", left_true)
    right_neighbor_true = gt.incident_edges_op(graph, "out", "max", right_true)
    any_true = graph.new_vertex_property(
        "bool", vals=(vprop.a | left_neighbor_true.a | right_neighbor_true.a)
    )
    return any_true
