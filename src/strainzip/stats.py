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
