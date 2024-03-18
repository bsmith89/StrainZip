import graph_tool as gt
import graph_tool.topology


def edge_has_no_siblings(g):
    "Check whether upstream or downstream sibling edges exist for every edge."
    vs = g.get_vertices()
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
        yield sort_order[sort_labels == i]
