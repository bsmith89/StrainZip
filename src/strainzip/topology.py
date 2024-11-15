import logging
from itertools import chain, product
from typing import List

import graph_tool as gt
import graph_tool.clustering
import graph_tool.topology
import numpy as np
from tqdm import tqdm


def edge_has_no_siblings(g):
    "Check whether upstream or downstream sibling edges exist for every edge."
    v_in_degree = g.degree_property_map("in")
    v_out_degree = g.degree_property_map("out")
    e_num_in_siblings = gt.edge_endpoint_property(g, v_in_degree, "target")
    e_num_out_siblings = gt.edge_endpoint_property(g, v_out_degree, "source")
    e_has_no_sibling_edges = g.new_edge_property(
        "bool", vals=(e_num_in_siblings.a <= 1) & (e_num_out_siblings.a <= 1)
    )
    return e_has_no_sibling_edges


from itertools import groupby


def prepare_unitig_graph(graph):
    no_sibling_edges = edge_has_no_siblings(graph)
    g_filt = gt.GraphView(graph, efilt=no_sibling_edges)

    # Find orphan vertices and filter them out.
    orphan_filt = g_filt.new_vertex_property(
        "bool", vals=g_filt.degree_property_map("total").a != 0
    )
    g_filt = gt.GraphView(g_filt, vfilt=orphan_filt)
    return g_filt


def iter_unitig_group_tables(unitig_graph, unitig_label):
    in_degree = unitig_graph.degree_property_map("in")
    out_degree = unitig_graph.degree_property_map("out")
    vertex_info_iter = sorted(
        zip(unitig_label.fa, in_degree.fa, out_degree.fa, unitig_graph.get_vertices()),
    )
    unitig_vertices_iter = groupby(vertex_info_iter, lambda x: x[0])

    data = list(
        sorted(
            zip(
                unitig_label.fa,
                in_degree.fa,
                out_degree.fa,
                unitig_graph.get_vertices(),
            )
        )
    )
    for k, g in groupby(vertex_info_iter, lambda x: x[0]):
        yield list(g)


def drop_in_edges_for_all(graph, vs):
    edge_target_vertex_property = gt.edge_endpoint_property(
        graph, graph.new_vertex_property("int", vals=graph.get_vertices()), "target"
    )
    edge_target_not_in_vertices = graph.new_edge_property("bool", val=1)
    gt.map_property_values(
        edge_target_vertex_property,
        edge_target_not_in_vertices,
        map_func=lambda x: x not in vs,
    )
    return gt.GraphView(graph, efilt=edge_target_not_in_vertices)


def iter_maximal_unitig_paths(graph):
    unitig_graph = prepare_unitig_graph(graph)
    unitig_labels, counts = gt.topology.label_components(unitig_graph, directed=False)

    # Cleave cycles.
    cycle_anchor_vertices = []
    for unitig_group_table in iter_unitig_group_tables(unitig_graph, unitig_labels):
        if len(unitig_group_table) == 1:
            cycle_anchor_vertices.append(unitig_group_table[0][-1])
        elif unitig_group_table[0][1] == 0:
            # The first entry in the (sorted) table has in-degree 0; it's a simple unitig.
            continue
        else:
            # The first entry in the (sorted) table does not have in-degree 0; it's a cycle.
            cycle_anchor_vertices.append(unitig_group_table[0][-1])

    # Sort vertices and yield labels
    unitig_graph_acyclic = drop_in_edges_for_all(unitig_graph, cycle_anchor_vertices)
    sort_order = np.argsort(gt.topology.topological_sort(unitig_graph_acyclic))
    vertex_list = unitig_graph_acyclic.get_vertices()
    for label, unitig_path_table in groupby(
        sorted(zip(unitig_labels.fa, sort_order, vertex_list)),
        key=lambda x: x[0],
    ):
        path = [x[-1] for x in unitig_path_table]
        # Paths of length 1 will happen for orphans or 1-cycles.
        if len(path) > 1:
            yield path


def find_tips(g, also_required=None):
    if also_required is None:
        also_required = True

    in_degree = g.degree_property_map("in")
    out_degree = g.degree_property_map("out")
    # np.where returns a tuple (maybe to deal with an N-dimensional mask?).
    result = np.where(((in_degree.a == 0) | (out_degree.a == 0)) & also_required)[0]
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


def get_shortest_distance(
    graph,
    roots: List[int],
    length=None,
    max_length=None,
    backlinked=None,
    verbose=False,
):
    # TODO: Write a test for this function:
    #   - Make sure that a single root is at distance 0 from itself.
    #   - Make sure that filtering doesn't break anything?
    #   - Check that _both_ in-neighbors and out-neighbors of the root are at a distance
    #     equal to the length of the root vertex.
    original_graph = graph
    if backlinked is not None:
        graph = backlinked
    else:
        graph = backlinked_graph(graph)

    # NOTE (2024-06-07): If the length property isn't cast to a float, then
    # (and therefore "edge lengths" are integers), the distances will also be integers
    # and vertices outside of max_dist (or in an unconnected component will have finite
    # distance (approximately the largest integer).
    if length is None:
        length = graph.new_vp("float", val=1.0)
    else:
        length = graph.new_vp("float", vals=graph.own_property(length).a)

    edge_length = gt.edge_endpoint_property(graph, length, "source")

    min_dist = np.inf * np.ones_like(graph.num_vertices(ignore_filter=True))
    for v in tqdm(roots, disable=(not verbose)):
        source_length = length[v]
        # Max length needs to be adjusted by the source vertex length.
        if max_length is not None:
            _max_length = max_length + source_length
        else:
            _max_length = None

        dist_to_v = gt.topology.shortest_distance(
            graph, v, weights=edge_length, directed=True, max_dist=_max_length
        )
        min_dist = np.minimum(min_dist, dist_to_v.a - source_length)

    return original_graph.new_vertex_property("float", vals=min_dist)


def get_shortest_distance_to_any_vertex(
    graph, roots, length=None, verbose=False, inplace=False
):
    if inplace:
        g = graph
    else:
        g = graph.copy()

    core_vertex = g.add_vertex()
    new_edges = [(core_vertex, v) for v in roots]
    g.add_edge_list(new_edges)
    shortest_dist = get_shortest_distance(
        g, roots=[core_vertex], length=length, verbose=verbose
    )
    g.remove_vertex(core_vertex)
    return graph.own_property(shortest_dist)


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


def build_bipartite_motif_graph(n, m):
    g = gt.Graph(product(range(n), range(n, n + m)))
    return g


def find_all_vertices_in_motifs(g, motif_graph):
    _, _, vertex_ids_prop = gt.clustering.motifs(
        g, len(motif_graph), motif_list=[motif_graph], return_maps=True
    )
    vertex_ids_prop = list(chain(*vertex_ids_prop))  # Flatten a nested list.
    return set(chain(*[list(motif_graph.own_property(vp).a) for vp in vertex_ids_prop]))


def find_blackbox_vertices(graph):
    vertices_in_blackboxes = find_all_vertices_in_motifs(
        graph,
        build_bipartite_motif_graph(2, 2),
    )
    return vertices_in_blackboxes


def find_self_looping_vertices(graph):
    vertices_with_self_loops = set(
        np.where(
            gt.incident_edges_op(
                graph,
                "in",
                "max",
                gt.generation.label_self_loops(graph, mark_only=True),
            ).a
            != 0
        )[0]
    )
    return vertices_with_self_loops
