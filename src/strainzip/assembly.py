import graph_tool as gt
import graph_tool.search
import graph_tool.topology
import numpy as np


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
    no_sibling_edges = edge_has_no_siblings(g)
    g_filt = gt.GraphView(g, efilt=no_sibling_edges, directed=True)
    labels, counts = gt.topology.label_components(g_filt, directed=False)
    return labels, counts, g_filt


def iter_maximal_unitig_paths(g):
    # FIXME: Fails when any of the unitigs are cycles.
    # Should be able to drop one edge from each cycle
    # in the GraphView and get the correct outputs...
    # Alternatively should be able to run gt.topology.all_circuits
    # and pre-select the circuits to yield as their own sorted lists.
    labels, counts, g_filt = label_maximal_unitigs(g)
    sort_order = gt.topology.topological_sort(g_filt)
    sort_labels = labels.a[sort_order]
    for i, _ in enumerate(counts):
        unitig = sort_order[sort_labels == i]
        if len(unitig) < 2:
            continue
        yield unitig


def find_tips(g):
    v_degree = g.degree_property_map("total")
    result = np.where(v_degree.a == 1)[
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
