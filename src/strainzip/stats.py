import graph_tool as gt
import numpy as np
import pandas as pd

from strainzip.topology import backlinked_graph


def num_unfiltered_nodes(graph):
    return graph.vp["filter"].a.sum()


def degree_stats(graph):
    out = pd.DataFrame(
        dict(
            in_degree=graph.get_in_degrees(graph.get_vertices()),
            out_degree=graph.get_out_degrees(graph.get_vertices()),
        )
    )
    return out.value_counts()


def total_kmers_counted(graph):
    vdepth = graph.vp["depth"].get_2d_array(pos=range(graph.gp["num_samples"])).sum(0)
    vlength = graph.vp["length"].a[graph.get_vertices()]
    return (vdepth * vlength).sum()


def mean_tig_length(graph):
    vlength = graph.vp["length"].a[graph.get_vertices()]
    return vlength.mean()


def depth_weighted_mean_tig_length(graph):
    vlength = graph.vp["length"].a[graph.get_vertices()]
    vdepth = graph.vp["depth"].get_2d_array(pos=range(graph.gp["num_samples"])).sum(0)
    return (vlength * vdepth).sum() / vdepth.sum()


def pairwise_distance_matrix(graph, vs):
    # FIXME: I think this matrix is indexed incorrectly. The output doesn't look like I expected.
    filt = graph.new_vertex_property(
        "bool", vals=pd.Index(graph.get_vertices()).to_series().isin(vs)
    )
    graph = gt.Graph(gt.GraphView(graph, vfilt=filt), prune=True)
    backlinked = backlinked_graph(graph)
    length = backlinked.own_property(graph.vp["length"])
    edge_weights = gt.edge_endpoint_property(backlinked, length, "source")
    shortest_dist = gt.topology.shortest_distance(
        backlinked, weights=edge_weights, directed=True
    )
    dmat = pd.DataFrame(
        shortest_dist.get_2d_array(pos=range(len(vs))),
        index=vs,
        columns=vs,
    )
    return dmat


def normalized_root_mean_squared_error(obs_depth_table, pred_depth_table):
    return (
        np.sqrt(np.square(obs_depth_table - pred_depth_table).sum().sum())
        / obs_depth_table.sum().sum()
    )
