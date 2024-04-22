import graph_tool as gt
import pandas as pd


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
