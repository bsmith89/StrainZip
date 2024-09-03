import logging
from multiprocessing import Pool as processPool

import graph_tool as gt
import numpy as np

import strainzip as sz
from strainzip.logging_util import phase_info, tqdm_debug

from ._base import App

DEFAULT_EPS = 1e-4


def _smooth_one_sample(arg):
    gt.openmp_set_num_threads(1)
    graph, sample_depth, sample_id, eps = arg

    logging.info(f"Starting smoothing sample {sample_id}.")
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
            type=float,
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
        self.parser.add_argument(
            "--sample-list",
            help="Only smooth a subset of samples. (Primarily useful for debugging.)",
        )

    def validate_and_transform_args(self, args):
        if args.sample_list:
            args.sample_list = [int(s) for s in args.sample_list.split(",")]
        return args

    def execute(self, args):
        with phase_info("Loading graph."):
            graph = sz.io.load_graph(args.inpath)
        with phase_info("Preparing depth property."):
            depth = gt.ungroup_vector_property(
                graph.vp["depth"].copy(), pos=range(graph.gp["num_samples"])
            )
            del graph.vp["depth"]
            graph.shrink_to_fit()

        if not args.sample_list:
            args.sample_list = list(range(graph.gp["num_samples"]))

        with phase_info("Starting parallel smoothing."), processPool(
            processes=args.processes
        ) as process_pool:
            depth_procs = process_pool.imap(
                _smooth_one_sample,
                (
                    (graph, depth[sample_id], sample_id, args.eps)
                    for sample_id in args.sample_list
                ),
            )

            depth_values = np.stack(
                [
                    d
                    for d in tqdm_debug(
                        depth_procs,
                        total=len(args.sample_list),
                        bar_format="{l_bar}{r_bar}",
                    )
                ]
            )

        graph.vp["depth"] = graph.new_vertex_property("vector<float>")
        graph.vp["depth"].set_2d_array(depth_values, pos=args.sample_list)

        sz.io.dump_graph(graph, args.outpath)
