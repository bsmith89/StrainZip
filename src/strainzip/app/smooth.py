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
        self.parser.add_argument(
            "--enforce-symmetry",
            action="store_true",
            help="Set reverse-complement depths to be the mean of both segments.",
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
                graph.vp["depth"], pos=range(graph.gp["num_samples"])
            )
            # NOTE (2024-09-03): By dropping the depth vertex property, we
            # avoid passing all this data around between processes. We'll add
            # this data back as an internal property at the end, before
            # writing the graph to disk.
            del graph.vp["depth"]
            # NOTE (2024-09-03): This method call seems necessary to get
            # graph_tool to release the memory.
            graph.shrink_to_fit()

        if not args.sample_list:
            args.sample_list = list(range(graph.gp["num_samples"]))

        with phase_info("Smoothing samples."), processPool(
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

        if args.enforce_symmetry:
            with phase_info("Enforcing depth symmetry."):

                def _rc_name(segment):
                    return segment[:-1] + {"+": "-", "-": "+"}[segment[-1]]

                # NOTE: Assumes that graph.vp['sequence'] is simple unitig+strand strings. No chaining.
                argsort_segments = np.argsort([s for s in graph.vp["sequence"]])
                argsort_rc_segments = np.argsort(
                    [_rc_name(s) for s in graph.vp["sequence"]]
                )
                # Using the argsort indices, sort the depth values by the names of the segments
                # as well as the names of the RC of the segments. Then take the mean of the two
                # arrays and invert the sorting.
                depth_values = (
                    (
                        depth_values[:, argsort_segments]
                        + depth_values[:, argsort_rc_segments]
                    )
                    / 2
                )[:, np.argsort(argsort_segments)]

        graph.vp["depth"] = graph.new_vertex_property("vector<float>")
        graph.vp["depth"].set_2d_array(depth_values, pos=args.sample_list)

        sz.io.dump_graph(graph, args.outpath)
