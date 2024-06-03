import logging

import pandas as pd
import scipy as sp
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

import strainzip as sz

from ._base import App

DEFAULT_NUM_PRECLUST = 5000
DEFAULT_EXPONENT = 1 / 2


class ClusterTigs(App):
    """Cluster graph vertices based on multi-sample depth profiles."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument(
            "thresh", type=float, help="Distance threshold for clustering."
        )
        self.parser.add_argument(
            "vertex_outpath", help="Outpath for mapping from vertices to clusters."
        )
        self.parser.add_argument(
            "segment_outpath", help="Outpath for mapping from segments to clusters."
        )
        self.parser.add_argument(
            "depth_outpath", help="Outpath for depth table for each cluster."
        )
        self.parser.add_argument(
            "shared_outpath",
            help="Outpath for unique, shared segments between pairs of clusters.",
        )
        self.parser.add_argument("meta_outpath", help="Outpath for cluster metadata.")
        self.parser.add_argument(
            "--num-preclust",
            type=int,
            default=DEFAULT_NUM_PRECLUST,
            help="Number of clusters to find during preclustering.",
        )
        self.parser.add_argument(
            "--exponent",
            type=float,
            default=DEFAULT_EXPONENT,
            help="Transform depths before clustering by raising to this exponent.",
        )
        self.parser.add_argument(
            "--random-seed",
            type=int,
            default=0,
            help="Random seed to initialize MiniBatchKMeans for preclustering.",
        )

    def execute(self, args):
        with sz.logging_util.phase_info("Loading graph"):
            graph = sz.io.load_graph(args.inpath)

        with sz.logging_util.phase_info("Tabulating vertex data"):
            vertex_depth = sz.results.depth_table(graph, graph.get_vertices()).T
            vertex_length = pd.Series(graph.vp["length"], index=graph.get_vertices())
            # Drop any zero-total-depth vertices
            num_zeros = (vertex_depth.sum(1) == 0).sum()
            logging.info(f"Dropping {num_zeros} 0-depth vertices.")
            vertex_depth, vertex_length = vertex_depth[lambda x: x.sum(1) > 0].align(
                vertex_length, axis=0, join="left"
            )

        with sz.logging_util.phase_info("Preclustering"):
            num_vertices, num_samples = vertex_depth.shape
            logging.info(
                f"Preclustering {num_vertices} tigs using depths across {num_samples} samples."
            )
            trsfm_vertex_depth = vertex_depth**args.exponent
            trsfm_vertex_depth_normalized = trsfm_vertex_depth.divide(
                (trsfm_vertex_depth ** (2)).sum(1) ** (1 / 2), axis=0
            )
            kmeans = MiniBatchKMeans(
                n_clusters=args.num_preclust, random_state=args.random_seed
            ).fit(trsfm_vertex_depth_normalized)
            with sz.logging_util.phase_info("Aggregating precluster stats"):
                vertex_preclust = pd.Series(kmeans.labels_, index=vertex_depth.index)
                preclust_length = vertex_length.groupby(vertex_preclust).sum()
                preclust_depth = (
                    (vertex_depth.multiply(vertex_length, axis=0))
                    .groupby(vertex_preclust)
                    .sum()
                    .divide(preclust_length, axis=0)
                )

        with sz.logging_util.phase_info("Clustering"):
            agglom = AgglomerativeClustering(
                n_clusters=None,  # pyright: ignore [reportArgumentType]
                metric="cosine",
                linkage="average",
                distance_threshold=args.thresh,
            ).fit(preclust_depth**args.exponent)
            with sz.logging_util.phase_info("Aggregating cluster stats"):
                clust = pd.Series(agglom.labels_, index=preclust_depth.index)
                clust_length = preclust_length.groupby(clust).sum()
                clust_depth = (
                    (preclust_depth.multiply(preclust_length, axis=0))
                    .groupby(clust)
                    .sum()
                    .divide(clust_length, axis=0)
                )
                vertex_clust = (
                    vertex_preclust.map(clust).rename_axis("vertex").rename("cluster")
                )
                vertex_segment = (
                    pd.Series(graph.vp["sequence"], index=graph.get_vertices())
                    .str.split(",")
                    .explode()
                    .rename("segment")
                    .rename_axis("vertex")
                    .reset_index()
                )
                segment_clust = vertex_segment.join(vertex_clust, on="vertex")[
                    ["segment", "cluster"]
                ]
                meta = pd.DataFrame(
                    dict(
                        num_vertices=vertex_clust.value_counts(),
                        num_segments=segment_clust.cluster.value_counts(),
                        num_unique_segments=segment_clust.drop_duplicates().cluster.value_counts(),
                        total_length=clust_length,
                        total_depth=clust_depth.sum(
                            1
                        ),  # Estimated cluster depth summed across all samples.
                    )
                ).rename_axis(index="cluster")

        with sz.logging_util.phase_info("Tallying shared segments"):
            segment_idx = (
                pd.Series(segment_clust.segment.unique(), name="segment")
                .rename_axis("segment_idx")
                .reset_index()
                .set_index(["segment"])
            )
            sparse_segment_clust_matrix = (
                segment_clust.drop_duplicates()
                .join(segment_idx, on="segment")
                .assign(tally=1)
            )
            sparse_segment_clust_matrix = sp.sparse.csc_array(
                (
                    sparse_segment_clust_matrix.tally,
                    (
                        sparse_segment_clust_matrix.segment_idx,
                        sparse_segment_clust_matrix.cluster,
                    ),
                )
            )
            # TODO (2024-05-30): Consider loading shared segments counts into the sparse, multi-indexed pd.Series without going through a dense matrix stage.
            shared_segments_matrix = pd.DataFrame(
                (sparse_segment_clust_matrix.T @ sparse_segment_clust_matrix).todense()
            ).rename_axis(index="clusterA", columns="clusterB")
            shared_segments = (
                shared_segments_matrix.stack()
                .to_frame(name="num_shared_segments")
                .reset_index()
                .assign(
                    num_clusterA_segments=lambda x: x.clusterA.map(
                        meta.num_unique_segments
                    ),
                    num_clusterB_segments=lambda x: x.clusterB.map(
                        meta.num_unique_segments
                    ),
                )[lambda x: (x.num_shared_segments > 0) & (x.clusterA < x.clusterB)]
            )

        with sz.logging_util.phase_info("Writing outputs"):
            vertex_clust.to_csv(args.vertex_outpath, sep="\t")
            segment_clust.drop_duplicates().to_csv(
                args.segment_outpath, sep="\t", index=False
            )
            clust_depth.to_csv(args.depth_outpath, sep="\t")
            shared_segments.to_csv(args.shared_outpath, sep="\t", index=False)
            meta.to_csv(args.meta_outpath, sep="\t")
