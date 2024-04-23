import logging
import multiprocessing

import graph_tool as gt
import numpy as np

import strainzip as sz

from ._base import App

DEFAULT_INERTIA = 0.5
DEFAULT_NUM_ITER = 50


def _load_graph_and_smooth_one_sample(arg):
    sample, inpath, inertia, num_iter, verbose = arg

    if verbose:
        logger = multiprocessing.log_to_stderr()
    else:
        logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)

    logger.info(f"Starting smoothing sample {sample}.")
    graph = sz.io.load_graph(inpath)
    sample_depth = gt.ungroup_vector_property(
        graph.vp["depth"], pos=range(graph.gp["num_samples"])
    )[sample]
    smoothed_sample_depth, _ = sz.flow.smooth_depth(
        graph,
        sample_depth,
        graph.vp["length"],
        inertia=inertia,
        num_iter=num_iter,
    )
    logger.info(f"Finished smoothing sample {sample}.")
    return smoothed_sample_depth.a


class SmoothDepths(App):
    """Smooth vertex depths and write output table."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("outpath")
        self.parser.add_argument(
            "--inertia",
            "-w",
            type=float,
            default=DEFAULT_INERTIA,
            help="Inertia parameter for depth smoothing.",
        )
        self.parser.add_argument(
            "--num-iter",
            "-n",
            type=int,
            default=DEFAULT_NUM_ITER,
            help="How many iterations for depth smoothing.",
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

        with multiprocessing.Pool(processes=args.processes) as pool:
            smoothed_depths = pool.map(
                _load_graph_and_smooth_one_sample,
                (
                    (
                        sample,
                        args.inpath,
                        args.inertia,
                        args.num_iter,
                        (args.verbose or args.debug),
                    )
                    for sample in range(graph.gp["num_samples"])
                ),
            )
        graph.vp["depth"].set_2d_array(
            np.stack(smoothed_depths), pos=range(graph.gp["num_samples"])
        )
        sz.io.dump_graph(graph, args.outpath)
