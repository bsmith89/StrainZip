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


def label_maximal_unitigs(g):
    "Assign unitig indices to vertices in maximal unitigs."
    with phase_debug("Filter out edges with siblings"):
        no_sibling_edges = edge_has_no_siblings(g)
        g_filt = gt.GraphView(g, efilt=no_sibling_edges, directed=True)
        logging.debug(g_filt)
    with phase_debug("Filter out orphans"):
        total_degree = (
            g_filt.degree_property_map("in").a + g_filt.degree_property_map("out").a
        )
        g_filt = gt.GraphView(g_filt, vfilt=total_degree != 0)
        logging.debug(g_filt)
    with phase_debug("Find distinct graph components"):
        # NOTE: Possibly non-deterministic??
        labels, counts = gt.topology.label_components(g_filt, directed=False)
    return labels, counts, g_filt


def iter_maximal_unitig_paths(g):
    logging.debug(g)
    # Filter edges and label components.
    labels, counts, g_filt = label_maximal_unitigs(g)
    logging.debug(g_filt)

    # Construct masks
    in_degree = g_filt.degree_property_map("in")
    out_degree = g_filt.degree_property_map("out")
    is_source = in_degree.fa == 0
    is_sink = out_degree.fa == 0
    vertex_ids = g_filt.get_vertices()
    source_set = set(vertex_ids[is_source])
    sink_set = set(vertex_ids[is_sink])

    with phase_debug("Iterating unitigs"):
        for i, _ in tqdm_debug(enumerate(counts), total=len(counts)):
            this_unitig_set = set(vertex_ids[labels.fa == i])

            # NOTE: Since we are filtering out orphans in label_maximal_unitigs
            # the only length-1 unitigs left are cycles.
            if len(this_unitig_set) == 1:
                # Not compressible.
                continue

            this_source_set = source_set & this_unitig_set
            if len(this_source_set) == 1:
                source = next(iter(this_source_set))
                this_sink_set = sink_set & this_unitig_set
                # NOTE: There should only one neighbor to the left, by construction.
                sink = next(iter(this_sink_set))
            elif len(this_source_set) == 0:
                # It must be a cycle.
                # Choose an arbitrary source vertex.
                source = next(iter(this_unitig_set))
                # Make the vertex to its left the sink.
                # NOTE: There should only one neighbor to the left, by construction.
                sink = g_filt.get_in_neighbors(source)[0]
            else:
                assert (
                    False
                ), "There should only be zero or one source vertex, by construction."

            # Find the path from source to sink
            unitig_path_list = list(gt.topology.all_paths(g_filt, source, sink))
            # NOTE: By construction, only one path should ever go between source and sink.
            yield unitig_path_list[0]


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
