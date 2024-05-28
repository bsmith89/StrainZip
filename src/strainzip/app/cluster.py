import pandas as pd
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

import strainzip as sz

from ._base import App

DEFAULT_NUM_PRECLUST = 5000


class ClusterTigs(App):
    """Cluster graph vertices based on multi-sample depth profiles."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument(
            "thresh", type=float, help="Distance threshold for clustering."
        )
        self.parser.add_argument(
            "cluster_outpath", help="Outpath for mapping from vertices to clusters."
        )
        self.parser.add_argument(
            "depth_outpath", help="Outpath for depth table for each cluster."
        )
        self.parser.add_argument("meta_outpath", help="Outpath for cluster metadata.")
        self.parser.add_argument(
            "--num-preclust",
            type=int,
            default=DEFAULT_NUM_PRECLUST,
            help="Number of clusters to find during preclustering.",
        )

    def execute(self, args):
        with sz.logging_util.phase_info("Loading graph"):
            graph = sz.io.load_graph(args.inpath)

        with sz.logging_util.phase_info("Tabulating vertex data"):
            vertex_depth = sz.results.depth_table(graph, graph.get_vertices()).T
            vertex_length = pd.Series(graph.vp["length"], index=graph.get_vertices())
            # Drop any zero-total-depth vertices
            vertex_depth, vertex_length = vertex_depth[lambda x: x.sum(1) > 0].align(
                vertex_length, axis=0, join="left"
            )

        with sz.logging_util.phase_info("Preclustering"):
            kmeans = MiniBatchKMeans(n_clusters=args.num_preclust).fit(vertex_depth)
            vertex_preclust = pd.Series(kmeans.labels_, index=vertex_depth.index)
            preclust_length = vertex_length.groupby(vertex_preclust).sum()
            preclust_depth = (vertex_depth * vertex_length).groupby(
                vertex_preclust
            ).sum() / preclust_length

        with sz.logging_util.phase_info("Clustering"):
            agglom = AgglomerativeClustering(
                n_clusters=None,  # pyright: ignore [reportArgumentType]
                metric="cosine",
                linkage="average",
                distance_threshold=args.thresh,
            ).fit(preclust_depth)
            clust = pd.Series(agglom.labels_, index=preclust_depth.index)
            clust_length = preclust_length.groupby(clust).sum()
            clust_depth = (preclust_depth * preclust_length).groupby(
                clust
            ).sum() / clust_length
            vertex_clust = vertex_preclust.map(clust)
            meta = pd.DataFrame(
                dict(
                    num_vertices=vertex_clust.value_counts(),
                    total_length=clust_length,
                    total_depth=clust_depth.sum(1),
                )
            )

        with sz.logging_util.phase_info("Writing outputs"):
            clust.to_csv(args.cluster_outpath, sep="\t", header=False)
            clust_depth.to_csv(args.depth_outpath, sep="\t")
            meta.to_csv(args.meta_outpath, sep="\t")
