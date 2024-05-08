import logging
import multiprocessing

import graph_tool as gt
import numpy as np

import strainzip as sz
from strainzip.logging_util import tqdm_debug

from ._base import App

DEFAULT_EPS = 1e-4


def _load_graph_and_smooth_one_sample(arg):
    gt.openmp_set_num_threads(1)
    graph, sample_id, eps = arg

    logging.info(f"Starting smoothing sample {sample_id}.")
    sample_depth = gt.ungroup_vector_property(graph.vp["depth"], pos=[sample_id])[0]
    smoothed_sample_depth, _ = sz.flow.smooth_depth(
        graph,
        sample_depth,
        graph.vp["length"],
        eps=eps,
    )
    logging.info(f"Finished smoothing sample {sample_id}.")
    return smoothed_sample_depth.fa


class SmoothDepths(App):
    """Smooth vertex depths and write output table."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("outpath")
        self.parser.add_argument(
            "--eps",
            "-e",
            type=int,
            default=DEFAULT_EPS,
            help="Stopping condition for depth smoothing. Smaller values will be more precise and take longer to run.",
        )
        self.parser.add_argument(
            "--processes",
            "-p",
            type=int,
            default=1,
            help="Number of parallel processes.",
        )

    def execute(self, args):
        graph = sz.io.load_graph(args.inpath)
        smoothed_depths = []

        with multiprocessing.Pool(processes=args.processes) as process_pool:
            depth_procs = process_pool.imap(
                _load_graph_and_smooth_one_sample,
                (
                    (graph, sample_id, args.eps)
                    for sample_id in range(graph.gp["num_samples"])
                ),
            )

            depth_values = np.stack(
                [
                    d
                    for d in tqdm_debug(
                        depth_procs,
                        total=graph.gp["num_samples"],
                        bar_format="{l_bar}{r_bar}",
                    )
                ]
            )

        graph.vp["depth"].set_2d_array(depth_values)

        sz.io.dump_graph(graph, args.outpath)
