import pandas as pd

import strainzip as sz

from ._base import App

COLUMN_ORDER = [
    "inpath",
    "kmer_length",
    "num_samples",
    "total_length",
    "num_vertices",
    "num_edges",
    "total_kmers",
    "depth_weighted_mean_tig_length",
]


class ShowGraphStats(App):
    """Gather sequences and depths from a graph."""

    def add_custom_cli_args(self):
        self.parser.add_argument(
            "graph_inpath", nargs="+", help="StrainZip formatted graph(s)."
        )

    def execute(self, args):
        graph_stats = {}
        print(*COLUMN_ORDER, sep="\t")
        for inpath in args.graph_inpath:
            graph = sz.io.load_graph(inpath)
            graph_stats[inpath] = pd.Series(
                dict(
                    inpath=inpath,
                    kmer_length=graph.gp["kmer_length"],
                    num_samples=graph.gp["num_samples"],
                    total_length=graph.vp["length"].a.sum(),
                    num_vertices=graph.num_vertices(),
                    num_edges=graph.num_edges(),
                    total_kmers=(
                        sz.results.total_depth_property(graph).a * graph.vp["length"].a
                    ).sum(),
                    depth_weighted_mean_tig_length=sz.stats.depth_weighted_mean_tig_length(
                        graph
                    ),
                )
            )
            print(*graph_stats[inpath][COLUMN_ORDER].values, sep="\t")

        # graph_stats = pd.DataFrame(graph_stats.values(), index=graph_stats.keys()).set_index('inpath')
