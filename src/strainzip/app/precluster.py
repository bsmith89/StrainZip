import logging

import pandas as pd
from sklearn.cluster import MiniBatchKMeans

import strainzip as sz

from ._base import App

DEFAULT_EXPONENT = 1


class PreClusterTigs(App):
    """Cluster graph vertices based on multi-sample depth profiles."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("num_preclust", type=int)
        self.parser.add_argument(
            "outpath", help="Outpath for mapping from vertices to clusters."
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
            vertex_preclust = pd.Series(
                kmeans.labels_, index=vertex_depth.index, name="cluster"
            ).rename_axis("vertex")

        with sz.logging_util.phase_info("Writing outputs"):
            vertex_preclust.to_csv(args.outpath, sep="\t")
