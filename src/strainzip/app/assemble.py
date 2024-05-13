import logging
from multiprocessing import Pool as ProcessPool

import graph_tool as gt
import numpy as np
import pandas as pd

import strainzip as sz
from strainzip.logging_util import phase_info

from ..depth_model import LogPlusAlphaLogNormal
from ..depth_model2 import SoftPlusNormal
from ..logging_util import tqdm_info
from ._base import App

DEFAULT_MAX_ROUNDS = 100
DEFAULT_OPT_MAXITER = 500
DEFAULT_CONDITION_THRESH = 1e6
DEFAULT_MIN_DEPTH = 0

DEPTH_MODELS = {
    "LogPlusAlphaLogNormal": (LogPlusAlphaLogNormal, dict(alpha=1.0)),
    "SoftPlusNormal": (SoftPlusNormal, dict()),
}

DEFAULT_DEPTH_MODEL = "LogPlusAlphaLogNormal"


def _estimate_flow(args):
    gt.openmp_set_num_threads(1)
    graph, sample_id = args
    depth = gt.ungroup_vector_property(graph.vp["depth"], pos=[sample_id])[0]
    length = graph.vp["length"]

    flow, _ = sz.flow.estimate_flow(
        graph,
        depth,
        length,
        eps=1e-6,
        maxiter=1000,
        flow_init=None,
        ifnotconverged="error",
    )
    # NOTE (2024-05-07): Because we're passing PropertyMaps between processes,
    # we need to be careful about the serialization and de-serialization.
    # For insance, flow.a contains a lot of 0s. If I tried to load
    # the vector<float> PropertyMap with these, it would silently truncate them.
    # to fit only the _unfiltered_ edges.
    return flow.fa


def _parallel_estimate_all_flows(graph, pool):
    flow_procs = pool.imap(
        _estimate_flow,
        (
            (
                graph,
                sample_id,
            )
            for sample_id in range(graph.gp["num_samples"])
        ),
    )

    # Collect rows of the flow table.
    flow_values = np.stack(
        [
            f
            for f in tqdm_info(
                flow_procs,
                total=graph.gp["num_samples"],
                bar_format="{l_bar}{r_bar}",
            )
        ]
    )
    flow = graph.new_edge_property("vector<float>")
    flow.set_2d_array(flow_values)
    return flow


def _iter_junction_deconvolution_data(junction_iter, graph, flow, max_paths):
    for j in junction_iter:
        in_neighbors = graph.get_in_neighbors(j)
        out_neighbors = graph.get_out_neighbors(j)
        n, m = len(in_neighbors), len(out_neighbors)
        if n * m > max_paths:
            continue

        # Collect flows
        in_flows = np.stack([flow[(i, j)] for i in in_neighbors])
        out_flows = np.stack([flow[(j, i)] for i in out_neighbors])

        # # FIXME (2024-04-20): Decide if I actually want to
        # # balance flows before fitting.
        # log_offset_ratio = np.log(in_flows.sum()) - np.log(out_flows.sum())
        # in_flows = np.exp(np.log(in_flows) - log_offset_ratio / 2)
        # out_flows = np.exp(np.log(out_flows) + log_offset_ratio / 2)

        yield j, in_neighbors, in_flows, out_neighbors, out_flows


def _calculate_junction_deconvolution(args):
    (
        junction,
        in_neighbors,
        in_flows,
        out_neighbors,
        out_flows,
        forward_stop,
        backward_stop,
        score_margin_thresh,
        condition_thresh,
        depth_model,
    ) = args
    n, m = len(in_neighbors), len(out_neighbors)
    try:
        fit, paths, named_paths, score_margin = sz.deconvolution.deconvolve_junction(
            in_neighbors,
            in_flows,
            out_neighbors,
            out_flows,
            model=depth_model,
            forward_stop=forward_stop,
            backward_stop=backward_stop,
        )
    except sz.errors.ConvergenceException:
        return (
            False,  # Convergence
            None,  # Score Margin
            None,  # Completeness
            None,  # Minimality
            None,  # Identifiability
            None,  # Result
        )

    X = sz.deconvolution.design_paths(n, m)[0]

    if not (score_margin > score_margin_thresh):
        return (
            True,  # Convergence
            False,  # Score Margin
            None,  # Completeness
            None,  # Minimality
            None,  # Identifiability
            None,  # Result
        )

    if not X[:, paths].sum(1).min() == 1:
        return (
            True,  # Convergence
            True,  # Score Margin
            False,  # Completeness
            None,  # Minimality
            None,  # Identifiability
            None,  # Result
        )

    if not len(paths) <= max(n, m):
        return (
            True,  # Convergence
            True,  # Score Margin
            True,  # Completeness
            False,  # Minimality
            None,  # Identifiability
            None,  # Result
        )

    try:
        condition = np.linalg.cond(fit.hessian_beta)
    except np.linalg.LinAlgError:
        return (
            True,  # Convergence
            True,  # Score Margin
            True,  # Completeness
            True,  # Minimality
            False,  # Identifiability
            None,  # Result
        )
    else:
        if not (condition < condition_thresh):
            return (
                True,  # Convergence
                True,  # Score Margin
                True,  # Completeness
                True,  # Minimality
                False,  # Identifiability
                None,  # Result
            )

    return (
        True,  # Convergence
        True,  # Score Margin
        True,  # Completeness
        True,  # Minimality
        True,  # Identifiability
        (junction, named_paths, {"path_depths": np.array(fit.beta.clip(0))}),  # Result
    )


def _parallel_calculate_junction_deconvolutions(
    junctions,
    graph,
    flow,
    depth_model,
    pool,
    forward_stop=0.0,
    backward_stop=0.0,
    score_margin_thresh=20.0,
    condition_thresh=1e5,
    max_paths=20,
):
    deconv_results = pool.imap_unordered(
        _calculate_junction_deconvolution,
        (
            (
                junction,
                in_neighbors,
                in_flows,
                out_neighbors,
                out_flows,
                forward_stop,
                backward_stop,
                score_margin_thresh,
                condition_thresh,
                depth_model,
            )
            for junction, in_neighbors, in_flows, out_neighbors, out_flows in _iter_junction_deconvolution_data(
                junctions, graph, flow, max_paths=max_paths
            )
        ),
    )

    batch = []
    postfix = dict(
        converged=0,
        best=0,
        complete=0,
        minimal=0,
        identifiable=0,
        split=0,
    )
    pbar = tqdm_info(
        deconv_results,
        total=len(junctions),
        bar_format="{l_bar}{r_bar}",
    )
    for (
        is_converged,
        is_best,
        is_complete,
        is_minimal,
        is_identifiable,
        result,
    ) in pbar:
        if is_converged:
            postfix["converged"] += 1
        if is_best:
            postfix["best"] += 1
        if is_complete:
            postfix["complete"] += 1
        if is_minimal:
            postfix["minimal"] += 1
        if is_identifiable:
            postfix["identifiable"] += 1
        if result is not None:
            postfix["split"] += 1
            junction, named_paths, path_depths_dict = result
            # print(f"{junction}: {named_paths}", end=" | ")
            batch.append(result)
        pbar.set_postfix(postfix)

    return batch


class DeconvolveGraph(App):
    """Run StrainZip graph deconvolution."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument(
            "score_thresh",
            type=float,
            help=(
                "BIC threshold to deconvolve a junction. "
                "Selected model must have a delta-BIC of greater than this amount"
            ),
        )
        self.parser.add_argument("outpath")
        self.parser.add_argument(
            "--min-depth",
            "-d",
            type=float,
            default=DEFAULT_MIN_DEPTH,
            help="Filter out edges with less than this minimum depth.",
        )
        self.parser.add_argument(
            "--max-rounds",
            "-n",
            type=int,
            default=DEFAULT_MAX_ROUNDS,
            help="Maximum rounds of graph deconvolution.",
        )
        self.parser.add_argument(
            "--condition-thresh",
            "-c",
            type=float,
            default=DEFAULT_CONDITION_THRESH,
            help=(
                "Maximum condition number of the Fisher Information Matrix to still deconvolve a junction. "
                " This is used as a proxy for how identifiable the depth estimates are."
            ),
        )
        self.parser.add_argument(
            "--model",
            dest="model_name",
            type=str,
            help="Which depth model to use.",
            choices=DEPTH_MODELS.keys(),
            default=DEFAULT_DEPTH_MODEL,
        )
        self.parser.add_argument(
            "--model-hyperparameters",
            type=str,
            nargs="+",
            metavar="KEY=VALUE",
            default=[],
            help=(
                "Value of the depth model hyperparameters in KEY=VALUE format. "
                "All VALUEs are assumed to be floats. "
                "Unassigned hyperparameters are given their default values."
            ),
        )
        self.parser.add_argument(
            "--opt-maxiter",
            type=int,
            default=DEFAULT_OPT_MAXITER,
            help="Run up to this number of optimization steps for regression convergence.",
        )
        self.parser.add_argument(
            "--processes",
            "-p",
            type=int,
            default=1,
            help="Number of parallel processes.",
        )
        self.parser.add_argument(
            "--no-prune",
            action="store_true",
            help="Keep filtered vertices instead of pruning them.",
        )

    def validate_and_transform_args(self, args):
        # Fetch model and default hyperparameters by name.
        depth_model_class, model_default_hyperparameters = DEPTH_MODELS[args.model_name]

        # Populate a dictionary of hyperparameters from the provided KEY=VALUE pairs.
        model_hyperparameters = {}
        for entry in args.model_hyperparameters:
            k, v = entry.split("=")
            assert (
                k in model_default_hyperparameters
            ), f"{k} does not appear to be a hyperparameter of {depth_model_class}."
            model_hyperparameters[k] = float(v)

        # Instantiate the depth model and assign it to args.
        args.depth_model = depth_model_class(
            maxiter=args.opt_maxiter,
            **(model_default_hyperparameters | model_hyperparameters),
        )

        return args

    def execute(self, args):
        gt.openmp_set_num_threads(args.processes)
        if args.debug or args.verbose:
            logging.getLogger("jax").setLevel(logging.CRITICAL)

        with phase_info("Loading graph"):
            graph = sz.io.load_graph(args.inpath)
            gm = sz.graph_manager.GraphManager(
                unzippers=[
                    sz.graph_manager.LengthUnzipper(),
                    sz.graph_manager.SequenceUnzipper(),
                    sz.graph_manager.VectorDepthUnzipper(),
                ],
                pressers=[
                    sz.graph_manager.LengthPresser(),
                    sz.graph_manager.SequencePresser(sep=","),
                    sz.graph_manager.VectorDepthPresser(),
                ],
            )
            gm.validate(graph)
            logging.info(
                f"Graph has {graph.num_vertices()} vertices and {graph.num_edges()} edges."
            )

        with ProcessPool(processes=args.processes) as process_pool:
            logging.info(
                f"Initialized multiprocessing pool with {args.processes} workers."
            )
            with phase_info("Pruning low-depth edges"):
                with phase_info("Optimizing flow"):
                    flow = _parallel_estimate_all_flows(
                        graph,
                        process_pool,
                    )
                # FIXME (2024-05-10): Confirm that setting vals from a get_2d_array has the right shape.
                not_low_depth_edge = graph.new_edge_property(
                    "bool",
                    vals=flow.get_2d_array(pos=range(graph.gp["num_samples"])).sum(0)
                    >= args.min_depth,
                )
                num_low_depth_edge = (not_low_depth_edge.a == 0).sum()
                logging.info(f"Filtering out {num_low_depth_edge} low-depth edges.")
                graph.set_edge_filter(not_low_depth_edge)

            with phase_info("Main loop"):
                for i in range(args.max_rounds):
                    logging.info(
                        f"Graph has {graph.num_vertices()} vertices and {graph.num_edges()} edges."
                    )
                    with phase_info(f"Round {i + 1}"):
                        with phase_info("Optimize flow"):
                            flow = _parallel_estimate_all_flows(
                                graph,
                                process_pool,
                            )
                        with phase_info("Finding junctions"):
                            junctions = sz.topology.find_junctions(
                                graph  # , also_required=consider.a
                            )
                            logging.debug(f"Found {len(junctions)} junctions.")
                        with phase_info("Optimizing junction deconvolutions"):
                            deconvolutions = _parallel_calculate_junction_deconvolutions(
                                junctions,
                                graph,
                                flow,
                                args.depth_model,
                                pool=process_pool,
                                forward_stop=0.0,
                                backward_stop=0.0,
                                score_margin_thresh=args.score_thresh,
                                condition_thresh=args.condition_thresh,
                                max_paths=100,  # FIXME: Consider whether I want this parameter at all.
                            )
                            # FIXME (2024-05-08): Sorting SHOULDN'T be (but is) necessary for deterministic unzipping.
                            deconvolutions = list(sorted(deconvolutions))
                        with phase_info("Unzipping junctions"):
                            new_unzipped_vertices = gm.batch_unzip(
                                graph, *deconvolutions
                            )
                            logging.info(
                                f"Unzipped junctions into {len(new_unzipped_vertices)} vertices."
                            )
                        with phase_info("Finding non-branching paths"):
                            unitig_paths = list(
                                sz.topology.iter_maximal_unitig_paths(graph)
                            )
                        with phase_info("Pressing tigs"):
                            new_pressed_vertices = gm.batch_press(
                                graph,
                                *[(path, {}) for path in unitig_paths],
                            )
                            logging.info(
                                f"Pressed non-branching paths into {len(new_pressed_vertices)} new tigs."
                            )
                        if len(new_unzipped_vertices) == 0:
                            logging.info("No junctions were unzipped. Stopping early.")
                            break
                else:
                    logging.info("Reached maximum number of deconvolution iterations.")

        with phase_info("Writing result."):
            logging.info(
                f"Graph has {graph.num_vertices()} vertices and {graph.num_edges()} edges."
            )
            sz.io.dump_graph(graph, args.outpath, prune=(not args.no_prune))
