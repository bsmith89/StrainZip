import logging

import pandas as pd

import strainzip as sz

from ._base import App


class DescribeGraph(App):
    """Gather sequences and depths from a graph."""

    def add_custom_cli_args(self):
        self.parser.add_argument(
            "graph_inpath", nargs="+", help="StrainZip formatted graph(s)."
        )

    def execute(self, args):
        degree_stats = {}
        depth_weighted_mean_tig_length = {}
        for inpath in args.graph_inpath:
            graph = sz.io.load_graph(inpath)
            results = sz.results.extract_vertex_data(graph)
            depth_weighted_mean_tig_length[
                inpath
            ] = sz.stats.depth_weighted_mean_tig_length(graph)
            degree_stats[inpath] = sz.stats.degree_stats(graph)
        depth_weighted_mean_tig_length = pd.Series(depth_weighted_mean_tig_length)
        degree_stats = pd.DataFrame(degree_stats).fillna(0).astype(int)
        print(depth_weighted_mean_tig_length.to_csv(sep="\t", header=False))
        print(degree_stats.to_csv(sep="\t"))
