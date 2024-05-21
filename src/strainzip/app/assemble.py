import logging
from multiprocessing import Pool as ProcessPool

import graph_tool as gt
import numpy as np

import strainzip as sz
from strainzip.logging_util import phase_info

from ..depth_model import LogPlusAlphaLogNormal
from ..depth_model2 import SoftPlusNormal
from ..logging_util import tqdm_info
from ._base import App

DEFAULT_MAX_ROUNDS = 100
DEFAULT_SCORE_THRESH = 10.0
DEFAULT_OPT_MAXITER = 10000
DEFAULT_RELATIVE_ERROR_THRESH = 0.1
DEFAULT_ABSOLUTE_ERROR_THRESH = 1.0
DEFAULT_MIN_DEPTH = 0

DEPTH_MODELS = {
    "LogPlusAlphaLogNormal": (LogPlusAlphaLogNormal, dict(alpha=1.0)),
    "SoftPlusNormal": (SoftPlusNormal, dict()),
}

DEFAULT_DEPTH_MODEL = "LogPlusAlphaLogNormal"


def _estimate_flow(args):
    # Limit each flow process to use just 1 core.
    gt.openmp_set_num_threads(1)

    graph, sample_id = args
    depth = gt.ungroup_vector_property(graph.vp["depth"], pos=[sample_id])[0]
    length = graph.vp["length"]

    # TODO (2024-05-14): This parallelization falls down because moving the graph
    # has too much serialization overhead, I think.
    # It's the same speed with maxiter=10 as maxiter=1000.
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
        chunksize=4,
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

        yield j, in_neighbors, in_flows, out_neighbors, out_flows


def _calculate_junction_deconvolution(args):
    (
        junction,
        in_neighbors,
        in_flows,
        out_neighbors,
        out_flows,
        depth_model,
    ) = args

    n, m = len(in_neighbors), len(out_neighbors)
    s = in_flows.shape[1]
    try:
        fit, paths, named_paths, score_margin = sz.deconvolution.deconvolve_junction(
            in_neighbors,
            in_flows,
            out_neighbors,
            out_flows,
            model=depth_model,
            exhaustive_thresh=50,
        )
    except sz.errors.ConvergenceException:
        return (
            False,
            np.nan,
            np.nan,
            np.nan,
            np.nan * np.empty(s),
            np.nan * np.empty(s),
            None,
        )

    if len(paths) == 0:
        return (
            True,
            0.0,
            0,
            0,
            np.nan * np.empty(s),
            np.nan * np.empty(s),
            None,
        )

    X = sz.deconvolution.design_all_paths(n, m)[0]
    excess_paths = len(paths) - max(n, m)
    completeness_ratio = (X[:, paths].sum(1) > 0).mean()
    relative_stderr = fit.stderr_beta / (np.abs(fit.beta) + 1)
    absolute_stderr = fit.stderr_beta

    return (
        True,
        score_margin,
        completeness_ratio,
        excess_paths,
        relative_stderr,
        absolute_stderr,
        (junction, named_paths, {"path_depths": np.array(fit.beta.clip(0))}),  # Result
    )


def _parallel_calculate_junction_deconvolutions(
    junctions,
    graph,
    flow,
    depth_model,
    pool,
    score_margin_thresh=20.0,
    relative_stderr_thresh=0.1,
    absolute_stderr_thresh=1.0,
    excess_thresh=1,
    completeness_thresh=1.0,
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
                depth_model,
            )
            for junction, in_neighbors, in_flows, out_neighbors, out_flows in _iter_junction_deconvolution_data(
                junctions, graph, flow, max_paths=max_paths
            )
        ),
        chunksize=20,
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
    pbar.set_postfix(postfix)
    for (
        is_converged,
        score_margin,
        completeness_ratio,
        excess_paths,
        relative_stderr,
        absolute_stderr,
        result,
    ) in pbar:
        if not is_converged:
            continue

        passes_identifiability = (
            (relative_stderr <= relative_stderr_thresh)
            | (absolute_stderr <= absolute_stderr_thresh)
        ).all()
        passes_score_margin = score_margin >= score_margin_thresh
        passes_excess = excess_paths <= excess_thresh
        passes_completeness = completeness_ratio >= completeness_thresh

        if is_converged:
            postfix["converged"] += 1
            if passes_score_margin:
                postfix["best"] += 1
            if passes_identifiability:
                postfix["identifiable"] += 1
            if passes_completeness:
                postfix["complete"] += 1
            if passes_excess:
                postfix["minimal"] += 1
        if (
            is_converged
            and passes_score_margin
            and passes_identifiability
            and passes_completeness
            and passes_excess
        ):
            postfix["split"] += 1

            batch.append(result)
        pbar.set_postfix(postfix, refresh=False)
        logging.debug(
            "{}\t{:.1f}\t{:.1f}\t{:d}\t{:.2f}\t{:.1f}".format(
                is_converged,
                score_margin,
                completeness_ratio,
                excess_paths,
                relative_stderr.max(),
                absolute_stderr.max(),
            )
        )

    return batch


class DeconvolveGraph(App):
    """Run StrainZip graph deconvolution."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument(
            "--score-thresh",
            type=float,
            default=DEFAULT_SCORE_THRESH,
            help=(
                "BIC threshold to deconvolve a junction. "
                "Selected model must have a delta-BIC of greater than this amount"
            ),
        )
        self.parser.add_argument("outpath")

        self.parser.add_argument(
            "--skip-drop-low-depth",
            action="store_true",
            help="Skip dropping of low-depth edges before deconvolution.",
        )
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
            "--relative-error-thresh",
            type=float,
            default=DEFAULT_RELATIVE_ERROR_THRESH,
            help=(
                "Maximum standard error to estimate ratio to still deconvolve a junction. "
                "This is used as a proxy for how identifiable the depth estimates are."
            ),
        )
        self.parser.add_argument(
            "--absolute-error-thresh",
            type=float,
            default=DEFAULT_ABSOLUTE_ERROR_THRESH,
            help=(
                "Relative error is not checked if the absolute error is less than this number. (To prevent very low-depth estimates from stopping deconvolution.)"
            ),
        )
        self.parser.add_argument(
            "--excess-thresh",
            type=int,
            default=0,
            help=("Acceptable over-abundance of paths in best deconvolution model."),
        )
        self.parser.add_argument(
            "--completeness-thresh",
            type=float,
            default=1.0,
            help=("TODO"),
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
        # # TODO (2024-05-15): After figuring out why purging is necessary,
        # # add this back to make it possible to keep filtered vertices/edges.
        # self.parser.add_argument(
        #     "--keep-filtered",
        #     action="store_true",
        #     help="Keep filtered vertices instead of removing them when saving the graph.",
        # )

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
        # TODO (2024-05-15): Limit each process to use just 1 core using threadpoolctl.

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

            if not args.skip_drop_low_depth:
                with phase_info("Pruning low-depth edges"):
                    with phase_info("Optimizing flow"):
                        flow = _parallel_estimate_all_flows(
                            graph,
                            process_pool,
                        )
                    # TODO (2024-05-10): Confirm that setting vals from a get_2d_array has the right shape.
                    not_low_depth_edge = (
                        flow.get_2d_array(pos=range(graph.gp["num_samples"])).sum(0)
                        >= args.min_depth
                    )
                    graph.ep["filter"] = graph.new_edge_property(
                        "bool", vals=not_low_depth_edge
                    )
                    num_low_depth_edge = (not_low_depth_edge == 0).sum()
                    logging.info(f"Filtering out {num_low_depth_edge} low-depth edges.")
                    graph.set_edge_filter(graph.ep["filter"])
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

            with phase_info("Main loop"):
                for i in range(args.max_rounds):
                    logging.info(
                        f"Graph has {graph.num_vertices()} vertices and {graph.num_edges()} edges."
                    )
                    with phase_info(f"Round {i + 1}"):
                        # TODO (2024-05-15): Why in the world do I need to purge filtered
                        # vertices/edges to get accurate flowing/deconvolution?
                        with phase_info("Dropping filtered vertices/edges"):
                            graph.purge_vertices()
                            graph.purge_edges()
                            # NOTE: This seems to be necessary to reactivate the embedded
                            # filter after purging:
                            graph.set_vertex_filter(graph.vp["filter"])
                            graph.set_edge_filter(graph.ep["filter"])
                        with phase_info("Optimize flow"):
                            flow = _parallel_estimate_all_flows(
                                graph,
                                process_pool,
                            )
                        with phase_info("Deconvolving junctions"):
                            deconvolutions = []
                            in_degree = graph.degree_property_map("in")
                            out_degree = graph.degree_property_map("out")
                            with phase_info("Safe junctions (Nx1 or 1xM)"):
                                is_safe_junction = (in_degree.a == 1) | (
                                    out_degree.a == 1
                                )
                                junctions_subset = sz.topology.find_junctions(
                                    graph, also_required=is_safe_junction
                                )
                                logging.info(
                                    f"Found {len(junctions_subset)} safe junctions"
                                )
                                deconvolutions_subset = _parallel_calculate_junction_deconvolutions(
                                    junctions_subset,
                                    graph,
                                    flow,
                                    args.depth_model,
                                    pool=process_pool,
                                    score_margin_thresh=args.score_thresh,
                                    relative_stderr_thresh=args.relative_error_thresh,
                                    absolute_stderr_thresh=args.absolute_error_thresh,
                                    excess_thresh=args.excess_thresh,
                                    completeness_thresh=args.completeness_thresh,
                                    max_paths=100,  # TODO (2024-05-01): Consider whether I want this parameter at all.
                                )
                                deconvolutions.extend(deconvolutions_subset)
                            with phase_info("Canonical junctions (2x2)"):
                                is_canonincal_junction = (in_degree.a == 2) & (
                                    out_degree.a == 2
                                )
                                junctions_subset = sz.topology.find_junctions(
                                    graph, also_required=is_canonincal_junction
                                )
                                logging.info(
                                    f"Found {len(junctions_subset)} canonical junctions"
                                )
                                deconvolutions_subset = _parallel_calculate_junction_deconvolutions(
                                    junctions_subset,
                                    graph,
                                    flow,
                                    args.depth_model,
                                    pool=process_pool,
                                    score_margin_thresh=args.score_thresh,
                                    relative_stderr_thresh=args.relative_error_thresh,
                                    absolute_stderr_thresh=args.absolute_error_thresh,
                                    excess_thresh=args.excess_thresh,
                                    completeness_thresh=args.completeness_thresh,
                                    max_paths=100,  # TODO (2024-05-01): Consider whether I want this parameter at all.
                                )
                                deconvolutions.extend(deconvolutions_subset)
                            with phase_info("Large junctions (> 2x2)"):
                                is_large_junction = (
                                    (in_degree.a >= 2) & (out_degree.a >= 2)
                                ) & ((in_degree.a + out_degree.a) > 4)
                                junctions_subset = sz.topology.find_junctions(
                                    graph, also_required=is_large_junction
                                )
                                logging.info(
                                    f"Found {len(junctions_subset)} large junctions"
                                )
                                deconvolutions_subset = _parallel_calculate_junction_deconvolutions(
                                    junctions_subset,
                                    graph,
                                    flow,
                                    args.depth_model,
                                    pool=process_pool,
                                    score_margin_thresh=args.score_thresh,
                                    relative_stderr_thresh=args.relative_error_thresh,
                                    absolute_stderr_thresh=args.absolute_error_thresh,
                                    excess_thresh=args.excess_thresh,
                                    completeness_thresh=args.completeness_thresh,
                                    max_paths=100,  # TODO (2024-05-01): Consider whether I want this parameter at all.
                                )
                                deconvolutions.extend(deconvolutions_subset)
                            # TODO (2024-05-08): Sorting SHOULDN'T be (but is) necessary for deterministic unzipping.
                            # TODO: (2024-05-16): Is this fixed now that I'm purging
                            # between each round (which I assume works because their was a
                            # problem with filtering)?
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
            sz.io.dump_graph(graph, args.outpath, purge=True)
