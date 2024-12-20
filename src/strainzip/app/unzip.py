import logging
import pickle
import warnings
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import List, Mapping, Optional, Tuple

import graph_tool as gt
import jax
import numpy as np
import pandas as pd

import strainzip as sz
from strainzip.logging_util import phase_info

from ..depth_model import NAMED_DEPTH_MODELS
from ..logging_util import tqdm_info
from ._base import App

DEFAULT_MAX_ROUNDS = 100
DEFAULT_SCORE = "bic"
DEFAULT_BALANCE_JUNCTIONS = True
DEFAULT_FLOW_SWAPPING = True
DEFAULT_SCORE_THRESH = 10.0
DEFAULT_RELATIVE_ERROR_THRESH = 0.1
DEFAULT_ABSOLUTE_ERROR_THRESH = 1.0
DEFAULT_MIN_DEPTH = 0
DEFAULT_EXTRA_LARGE_THRESH = 100

DEFAULT_DEPTH_MODEL = "Default"


@dataclass
class DeconvolutionProblem:
    junction: int
    junction_depth: float
    in_neighbors: List[int]
    out_neighbors: List[int]
    in_flows: np.ndarray
    out_flows: np.ndarray


@dataclass
class DeconvolutionResult:
    converged: bool
    score_margin: float
    completeness_ratio: float
    excess_paths: int
    relative_stderr: np.ndarray
    absolute_stderr: np.ndarray
    unzip: Optional[Tuple[int, List[Tuple[int, int]], Mapping[str, np.ndarray]]]
    fit: Optional[sz.depth_model.DepthModelResult]


def _run_drop_low_depth_edges(graph, gm, min_depth, mapping_func):
    with phase_info("Pruning low-depth edges"):
        flow = _run_estimate_all_flows(
            graph,
            mapping_func=mapping_func,  # partial(process_pool.imap, chunksize=4),
        )
        # TODO (2024-05-10): Confirm that setting vals from a get_2d_array has the right shape.
        not_low_depth_edge = (
            flow.get_2d_array(pos=range(graph.gp["num_samples"])).sum(0) >= min_depth
        )
        num_low_depth_edge = (not_low_depth_edge == 0).sum()
        logging.info(f"Filtering out {num_low_depth_edge} low-depth edges.")
        graph.ep["filter"].a = not_low_depth_edge
        num_low_depth_edge = (not_low_depth_edge == 0).sum()
        _run_press_unitigs(graph, gm)


def _run_press_unitigs(graph, graph_manager):
    with phase_info("Pressing unitigs"):
        with phase_info("Finding non-branching paths"):
            unitig_paths = list(sz.topology.iter_maximal_unitig_paths(graph))
        with phase_info("Compressing paths"):
            new_pressed_vertices = graph_manager.batch_press(
                graph,
                *[(path, {}) for path in unitig_paths],
            )
            logging.info(
                f"Pressed non-branching paths into {len(new_pressed_vertices)} new tigs."
            )


def _estimate_flow(args):
    # Limit each flow process to use just 1 core.
    gt.openmp_set_num_threads(1)

    graph, depth = args
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
    # For instance, flow.a contains a lot of 0s. If I tried to load
    # the vector<float> PropertyMap with these, it would silently truncate them.
    # to fit only the _unfiltered_ edges.
    return flow.fa


def _run_estimate_all_flows(graph, mapping_func):
    with phase_info("Optimize flow"):
        # NOTE (2024-09-07): Making a copy of graph missing the depth matrix to
        # save space when passing it among processes.
        # This is kinda silly, but works for memory management. It's dumb though.
        _graph = graph.copy()
        depth = _graph.vp["depth"]
        del _graph.vp["depth"]
        _graph.shrink_to_fit()

        flow_procs = mapping_func(
            _estimate_flow,
            (
                (
                    _graph,
                    gt.ungroup_vector_property(depth, pos=[sample_id])[0],
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


def _iter_junction_deconvolution_problems(junction_iter, graph, depth, flow):
    for j in junction_iter:
        in_neighbors = graph.get_in_neighbors(j)
        out_neighbors = graph.get_out_neighbors(j)
        n, m = len(in_neighbors), len(out_neighbors)

        # Collect flows
        in_flows = np.stack([flow[(i, j)] for i in in_neighbors])
        out_flows = np.stack([flow[(j, i)] for i in out_neighbors])

        junction_depth = depth[j]

        yield DeconvolutionProblem(
            j, junction_depth, in_neighbors, out_neighbors, in_flows, out_flows
        )


def _decide_if_flow_ordering_swap(flowsA, flowsB):
    if flowsA.shape[0] > flowsB.shape[0]:
        return True
    elif flowsB.shape[0] > flowsA.shape[0]:
        return False
    elif flowsA.sum() > flowsB.sum():
        return True
    elif flowsB.sum() > flowsA.sum():
        return False
    elif np.sign(flowsA - flowsB).sum() > 0:
        return True
    elif np.sign(flowsB - flowsA).sum() > 0:
        return False
    # elif flowsB > flowsA earlier in the list than flowsB > flowsA. In other
    # In other words: the first entry (reading in an arbitrary order) where
    # flowA > flowB or flowB > flowA, which one is greater?
    elif (flowsA == flowsB).all():
        # Flows are identical. Doesn't matter what order.
        return False
    else:
        warnings.warn("Cannot decide on an ordering in/out flows for this junction.")


def _calculate_junction_deconvolution(args):
    # # TODO (2024-05-15): Limit each process to use just 1 core using threadpoolctl?
    # NOTE (2024-06-03): Setting jaxopt to use 64-bit floats should increase the convergence rate.
    jax.config.update("jax_enable_x64", True)
    # TODO (2024-06-03): Also consider JAX persistent compilation cache,
    # although there's a risk of deadlocking with multiprocessing code.

    (
        deconv_problem,
        depth_model,
        score_name,
        balance,
        swap,
    ) = args

    in_neighbors = deconv_problem.in_neighbors
    out_neighbors = deconv_problem.out_neighbors
    in_flows = deconv_problem.in_flows
    out_flows = deconv_problem.out_flows
    junction = deconv_problem.junction
    junction_depth = deconv_problem.junction_depth

    # NOTE (2024-06-03): Here, with the intent of treating both forward and reverse
    # version of each junction identically, I'm picking a "canonical ordering"
    # of in-flows and out-flows.
    # I'll then conditionally reverse all the paths coming out the other side.
    # NOTE (2024-11-14): I've decided this is futile as junction symmetry is
    # broken for entirely different reasons.
    if swap:
        do_swap = _decide_if_flow_ordering_swap(
            deconv_problem.in_flows, deconv_problem.out_flows
        )
    else:
        do_swap = False

    if do_swap:
        # Swap everything
        in_neighbors, out_neighbors = out_neighbors, in_neighbors
        in_flows, out_flows = out_flows, in_flows

    if balance:
        # NOTE (2024-06-06): Balance in and out flows.
        # This (greatly) decreases the error, and therefore
        # the estimated stderrs of the path depth estimates.
        # As a result, far more junctions are deconvolved. However,
        # it's possible that this error was a signal that
        # the depths are difficult to estimate.
        # So the overall quality of the junctions that are now deconvolved
        # (and weren't before) is not obvious.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="divide by zero encountered in divide",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in divide",
            )
            # TODO (2024-06-10): Figure out why I get two different warnings here.
            in_flows_adjusted = in_flows * np.nan_to_num(
                junction_depth / in_flows.sum(axis=0), nan=1
            )
            out_flows_adjusted = out_flows * np.nan_to_num(
                junction_depth / out_flows.sum(axis=0), nan=1
            )
            # NOTE: nan_to_num call simply corrects samples with no depth and
            # no in/out flow with a factor of 1.0; this will have no effect on
            # those samples, since their depth is 0.
    else:
        in_flows_adjusted = in_flows
        out_flows_adjusted = out_flows

    n, m = len(in_neighbors), len(out_neighbors)
    (
        fit,
        paths,
        named_paths,
        score_margin,
    ) = sz.deconvolution.deconvolve_junction_exhaustive(
        in_neighbors,
        in_flows_adjusted,
        out_neighbors,
        out_flows_adjusted,
        model=depth_model,
        score_name=score_name,
    )

    if do_swap:
        # We swapped in and out above, so we'll swap them back here.
        named_paths = [(j, i) for i, j in named_paths]

    X = sz.deconvolution.design_all_paths(n, m)[0]
    excess_paths = len(paths) - max(n, m)
    completeness_ratio = (X[:, paths].sum(1) > 0).mean()
    relative_stderr = fit.stderr_beta / (np.abs(fit.beta) + 1)
    absolute_stderr = fit.stderr_beta

    unzip = (
        junction,
        named_paths,
        {"path_depths": np.array(fit.beta.clip(min=0))},
    )

    result = DeconvolutionResult(
        converged=fit.converged,
        score_margin=score_margin,
        completeness_ratio=completeness_ratio,
        excess_paths=excess_paths,
        relative_stderr=relative_stderr,
        absolute_stderr=absolute_stderr,
        unzip=unzip,
        fit=fit,
    )

    return result


def _test_identifiability(deconv, relative_stderr_thresh, absolute_stderr_thresh):
    return (
        (deconv.relative_stderr <= relative_stderr_thresh)
        | (deconv.absolute_stderr <= absolute_stderr_thresh)
    ).all()


def _run_calculate_junction_deconvolutions(
    deconv_problems,
    depth_model,
    mapping_func,
    balance=True,
    swap=True,
    score_name="bic",
    score_margin_thresh=20.0,
    relative_stderr_thresh=0.1,
    absolute_stderr_thresh=1.0,
    excess_thresh=1,
    completeness_thresh=1.0,
):
    num_problems = len(deconv_problems)

    results_iter = mapping_func(
        _calculate_junction_deconvolution,
        (
            (problem, depth_model, score_name, balance, swap)
            for problem in deconv_problems
        ),
    )

    postfix = dict(
        converged=0,
        best=0,
        complete=0,
        minimal=0,
        identifiable=0,
        split=0,
    )
    pbar = tqdm_info(
        results_iter,
        total=num_problems,
        bar_format="{l_bar}{r_bar}",
    )
    pbar.set_postfix(postfix)
    unzip_batch = []
    all_results = []
    for deconv in pbar:
        all_results.append(deconv)
        if not deconv.converged:
            continue

        passes_converged = deconv.converged
        passes_identifiability = _test_identifiability(
            deconv, relative_stderr_thresh, absolute_stderr_thresh
        )
        passes_score_margin = deconv.score_margin >= score_margin_thresh
        passes_excess = deconv.excess_paths <= excess_thresh
        passes_completeness = deconv.completeness_ratio >= completeness_thresh

        if passes_converged:
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
            passes_converged
            and passes_score_margin
            and passes_identifiability
            and passes_completeness
            and passes_excess
        ):
            postfix["split"] += 1
            unzip_batch.append(deconv.unzip)
        pbar.set_postfix(postfix, refresh=False)

        # # FIXME (2024-06-02): relative/absolute_stderr may both be empty arrays (if the deconvolution resulted in 0 paths).
        # logging.debug(
        #     "{}\t{:.1f}\t{:.1f}\t{:d}\t{:.2f}\t{:.1f}".format(
        #         deconv.converged,
        #         deconv.score_margin,
        #         deconv.completeness_ratio,
        #         deconv.excess_paths,
        #         deconv.relative_stderr.max(),
        #         deconv.absolute_stderr.max(),
        #     )
        # )

    return unzip_batch, all_results


class UnzipGraph(App):
    """Run StrainZip graph deconvolution."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("outpath")
        self.parser.add_argument(
            "--checkpoint-dir",
            help="Write graph to checkpoint directory before each round of deconvolution.",
        )
        self.parser.add_argument(
            "--score-thresh",
            type=float,
            default=DEFAULT_SCORE_THRESH,
            help=(
                "BIC threshold to deconvolve a junction. "
                "Selected model must have a delta-BIC of greater than this amount"
            ),
        )
        self.parser.add_argument(
            "--score",
            dest="score_name",
            default=DEFAULT_SCORE,
            choices=["bic", "aic", "aicc"],
        )
        self.parser.add_argument(
            "--no-balance",
            action="store_true",
            default=(not DEFAULT_BALANCE_JUNCTIONS),
            help="Whether or not to balance total in and out depths at junctions during deconvolution.",
        )
        self.parser.add_argument(
            "--no-swapping",
            action="store_true",
            default=(not DEFAULT_FLOW_SWAPPING),
            help="Whether or not to swap in and out flows at junctions during deconvolution in order to synchronize reverse-complements.",
        )
        self.parser.add_argument(
            "--no-drop-low-depth",
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
            "--skip-safe",
            action="store_true",
            help="Don't try to deconvolve safe junctions.",
        )
        self.parser.add_argument(
            "--skip-canonical",
            action="store_true",
            help="Don't try to deconvolve canonical junctions.",
        )
        self.parser.add_argument(
            "--skip-large",
            action="store_true",
            help="Don't try to deconvolve large junctions.",
        )
        self.parser.add_argument(
            "--skip-extra-large",
            dest="skip_extralarge",
            action="store_true",
            help="Don't try to deconvolve extra large junctions.",
        )
        self.parser.add_argument(
            "--extra-large",
            type=int,
            default=DEFAULT_EXTRA_LARGE_THRESH,
            help="Junction size to consider 'extra large'.",
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
            choices=NAMED_DEPTH_MODELS.keys(),
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
            "--processes",
            "-p",
            type=int,
            default=1,
            help="Number of parallel processes.",
        )

    def validate_and_transform_args(self, args):
        # Fetch model and default hyperparameters by name.
        depth_model_class, model_default_hyperparameters = NAMED_DEPTH_MODELS[
            args.model_name
        ]

        # Populate a dictionary of hyperparameters from the provided KEY=VALUE pairs.
        model_hyperparameters = {}
        for entry in args.model_hyperparameters:
            k, v = entry.split("=")
            model_hyperparameters[k] = float(v)

        # Instantiate the depth model and assign it to args.
        args.depth_model = depth_model_class(
            **(model_default_hyperparameters | model_hyperparameters),
        )

        return args

    def execute(self, args):
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
            # NOTE: The edge_property "filter" is un-managed and un-validated.
            graph.ep["filter"] = graph.new_edge_property("bool", val=1)
            graph.set_edge_filter(graph.ep["filter"])
            gm.validate(graph)
            logging.info(
                f"Graph has {graph.num_vertices()} vertices and {graph.num_edges()} edges."
            )

        with ProcessPool(
            processes=args.processes,
            # maxtasksperchild=10_000,  # TODO: (2024-09-13) Figure out if this helps with the processes dying for the super big graph.
        ) as process_pool:
            logging.info(
                f"Initialized multiprocessing pool with {args.processes} workers."
            )

            if not args.no_drop_low_depth:
                _run_drop_low_depth_edges(
                    graph,
                    gm,
                    args.min_depth,
                    mapping_func=partial(process_pool.imap_unordered, chunksize=4),
                )
            with phase_info("Main loop"):
                logging.info(f"Deconvolving junctions with {args.depth_model}.")
                for i in range(args.max_rounds):
                    logging.info(
                        f"Graph has {graph.num_vertices()} vertices and {graph.num_edges()} edges."
                    )
                    with phase_info(f"Round {i+1}"):
                        # TODO (2024-05-15): Why in the world do I need to purge filtered
                        # vertices/edges to get accurate flowing/deconvolution?
                        with phase_info("Dropping filtered vertices/edges"):
                            graph.purge_vertices()
                            graph.purge_edges()
                            # NOTE: This seems to be necessary to reactivate the embedded
                            # filter after purging:
                            graph.set_vertex_filter(graph.vp["filter"])
                            graph.set_edge_filter(graph.ep["filter"])
                        if args.checkpoint_dir:
                            with phase_info("Checkpointing graph"):
                                sz.io.dump_graph(
                                    graph,
                                    f"{args.checkpoint_dir}/checkpoint_{i}.sz",
                                )
                        flow = _run_estimate_all_flows(
                            graph,
                            partial(process_pool.imap, chunksize=4),
                        )

                        with phase_info("Finding unidentifiable flows"):
                            # TODO: Be more confident about this.
                            # NOTE (2024-06-12): I've just added this because I
                            # realized that bipartite subgraphs create
                            # inevitable uncertainty about the flows along their
                            # edges. Since the flow is unidentifiable, the
                            # deconvolution is also unidentifiable.
                            # Here I'm searching for a "2x2" motif, where
                            # two vertices both point at two other vertices.
                            # I believe this motif should show up in ALL
                            # such unidentifiable flow situations, (including
                            # analogous 3x3 situatations).
                            vertices_in_blackboxes = sz.topology.find_blackbox_vertices(
                                graph
                            )
                            logging.info(
                                f"Found {len(vertices_in_blackboxes)} vertices in 'black-boxes'."
                            )
                            # NOTE (2024-06-19): I've now also realized that
                            # self-looping situations are similarly tricky to
                            # estimate flow. I guess you might call them
                            # partially unidentifiable, since the flow on the
                            # looping edge is only as well specified as the
                            # out-flows from the vertex are well specified.
                            # As a result, a set of connect-and-self-looping
                            # vertices (i.e. a dumbell: O-O),
                            # would also be unidentifiable flows, since we
                            # don't know how much flow is on the self-looping
                            # edges and how much is on the connecting edge.
                            # Unlike black-boxes, not ALL self-looping vertices
                            # are necessarily unidentifiable, and the "dumbell"
                            # motif doesn't capture all such issues. However,
                            # I'm excluding junctions with self-looping entirely
                            # from deconvolution. These motifs also have other
                            # issues with deconvolution (since they can pop out
                            # incorrectly and there's no way to prevent this).
                            vertices_with_self_loops = (
                                sz.topology.find_self_looping_vertices(graph)
                            )
                            logging.info(
                                f"Found {len(vertices_with_self_loops)} vertices with self-loops."
                            )
                            vertices_with_unidentifiable_flows = (
                                vertices_in_blackboxes | vertices_with_self_loops
                            )
                            logging.info(
                                f"Excluding all {len(vertices_with_unidentifiable_flows)} vertices with unidentifiable flows."
                            )

                        with phase_info("Deconvolving junctions"):
                            deconvolutions = []
                            in_degree = graph.degree_property_map("in")
                            out_degree = graph.degree_property_map("out")

                            junction_sets = [
                                (
                                    "safe",  # name
                                    "Safe junctions (Nx1 or 1xM)",  # Phase label
                                    (in_degree.a == 1)
                                    | (out_degree.a == 1),  # indicator
                                    args.skip_safe,  # skip_flag
                                ),
                                (
                                    "canonical",  # name
                                    "Canonical junctions (2x2)",  # Phase label
                                    (in_degree.a == 2)
                                    & (out_degree.a == 2),  # indicator
                                    args.skip_canonical,  # skip_flag
                                ),
                                (
                                    "large",  # name
                                    f"Large junctions (<={args.extra_large} minimal, complete pathsets)",  # Phase label
                                    (
                                        ((in_degree.a >= 2) & (out_degree.a >= 2))
                                        & (in_degree.a + out_degree.a > 4)
                                        & (
                                            sz.deconvolution.num_minimal_complete_pathsets(
                                                in_degree.a, out_degree.a
                                            )
                                            <= args.extra_large
                                        )
                                    ),  # indicator
                                    args.skip_large,  # skip_flag
                                ),
                                (
                                    "extralarge",  # name
                                    f"Extra-large junctions (>{args.extra_large} minimal, complete pathsets)",  # Phase label
                                    (
                                        ((in_degree.a >= 2) & (out_degree.a >= 2))
                                        & (in_degree.a + out_degree.a > 4)
                                        & (
                                            sz.deconvolution.num_minimal_complete_pathsets(
                                                in_degree.a, out_degree.a
                                            )
                                            > args.extra_large
                                        )
                                    ),  # indicator
                                    args.skip_extralarge,  # skip_flag
                                ),
                            ]

                            for (
                                junction_set_name,
                                phase_label,
                                indicator,
                                skip_flag,
                            ) in junction_sets:
                                with phase_info(phase_label):
                                    junctions_subset = set(
                                        sz.topology.find_junctions(
                                            graph, also_required=indicator
                                        )
                                    )
                                    logging.info(
                                        f"Found {len(junctions_subset)} {junction_set_name} junctions."
                                    )
                                    num_blackbox_junctions = len(
                                        set(junctions_subset)
                                        & vertices_with_unidentifiable_flows
                                    )
                                    logging.info(
                                        f"Of these, {num_blackbox_junctions} have unidentifiable flows."
                                    )
                                    junctions_subset = junctions_subset - set(
                                        vertices_with_unidentifiable_flows
                                    )
                                    # NOTE (2024-11-14): To make checkpointing and logging easier (so not good reasons)
                                    # I fully load this sequence ahead of time.
                                    # This has performance implications.
                                    deconv_problems_subset = list(
                                        _iter_junction_deconvolution_problems(
                                            junctions_subset,
                                            graph,
                                            graph.vp["depth"],
                                            flow,
                                        )
                                    )
                                    if args.checkpoint_dir:
                                        with phase_info("Checkpointing deconvolutions"):
                                            with open(
                                                f"{args.checkpoint_dir}/junctions_{junction_set_name}_{i+1}.pkl",
                                                "wb",
                                            ) as f:
                                                pickle.dump(deconv_problems_subset, f)
                                    if not skip_flag:
                                        (
                                            deconv_results_subset,
                                            _,
                                        ) = _run_calculate_junction_deconvolutions(
                                            deconv_problems_subset,
                                            args.depth_model,
                                            balance=(not args.no_balance),
                                            swap=(not args.no_swapping),
                                            mapping_func=partial(
                                                process_pool.imap_unordered, chunksize=1
                                            ),
                                            score_name=args.score_name,
                                            score_margin_thresh=args.score_thresh,
                                            relative_stderr_thresh=args.relative_error_thresh,
                                            absolute_stderr_thresh=args.absolute_error_thresh,
                                            excess_thresh=args.excess_thresh,
                                            completeness_thresh=args.completeness_thresh,
                                        )
                                        deconvolutions.extend(deconv_results_subset)
                                    else:
                                        logging.info(
                                            "Skipping {junction_set_name} junctions."
                                        )

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
                        _run_press_unitigs(graph, gm)
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


class BenchmarkDepthModel(App):
    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="Checkpointed junctions in pkl format.")
        self.parser.add_argument("outpath")
        self.parser.add_argument(
            "--score-thresh",
            type=float,
            default=DEFAULT_SCORE_THRESH,
            help=(
                "BIC threshold to deconvolve a junction. "
                "Selected model must have a delta-BIC of greater than this amount"
            ),
        )
        self.parser.add_argument(
            "--score",
            dest="score_name",
            default=DEFAULT_SCORE,
            choices=["bic", "aic", "aicc"],
        )
        self.parser.add_argument(
            "--no-balance",
            action="store_true",
            default=(not DEFAULT_BALANCE_JUNCTIONS),
            help="Whether or not to balance total in and out depths at junctions during deconvolution.",
        )
        self.parser.add_argument(
            "--no-swapping",
            action="store_true",
            default=(not DEFAULT_FLOW_SWAPPING),
            help="Whether or not to swap in and out flows at junctions during deconvolution in order to synchronize reverse-complements.",
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
            choices=NAMED_DEPTH_MODELS.keys(),
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
            "--processes",
            "-p",
            type=int,
            default=1,
            help="Number of parallel processes.",
        )
        self.parser.add_argument(
            "--slice",
            help=(
                "Only run on a subset of problems. "
                "Specified as a `LEFT-INCLUSIVE:RIGHT-EXCLUSIVE` argument."
            ),
        )

    def validate_and_transform_args(self, args):
        # Fetch model and default hyperparameters by name.
        depth_model_class, model_default_hyperparameters = NAMED_DEPTH_MODELS[
            args.model_name
        ]

        model_hyperparameters = {}
        for entry in args.model_hyperparameters:
            k, v = entry.split("=")
            model_hyperparameters[k] = float(v)

        # Instantiate the depth model with hyperparameter and assign it to args.
        args.depth_model = depth_model_class(
            **(model_default_hyperparameters | model_hyperparameters),
        )

        if args.slice is None:
            args.slice = slice(None)
        else:
            args.slice = slice(*[int(a) for a in args.slice.split(":")])

        return args

    def execute(self, args):
        if args.debug or args.verbose:
            logging.getLogger("jax").setLevel(logging.CRITICAL)

        with phase_info("Loading test cases"):
            with open(args.inpath, "rb") as f:
                deconv_problems = pickle.load(f)
                # Subset up here so that we don't need to keep the whole set in
                # memory.
                deconv_problems = deconv_problems[args.slice]
            logging.info(f"Loaded {len(deconv_problems)} problems.")

        with ProcessPool(processes=args.processes) as process_pool:
            logging.info(
                f"Initialized multiprocessing pool with {args.processes} workers."
            )
            logging.info(f"Deconvolving junctions with {args.depth_model}.")

            batch_unzip, all_results = _run_calculate_junction_deconvolutions(
                deconv_problems,
                args.depth_model,
                balance=(not args.no_balance),
                swap=(not args.no_swapping),
                mapping_func=partial(process_pool.imap_unordered, chunksize=40),
                score_name=args.score_name,
                score_margin_thresh=args.score_thresh,
                relative_stderr_thresh=args.relative_error_thresh,
                absolute_stderr_thresh=args.absolute_error_thresh,
                excess_thresh=args.excess_thresh,
                completeness_thresh=args.completeness_thresh,
            )

        results = []
        for res in all_results:
            passes_identifiability = _test_identifiability(
                res, args.relative_error_thresh, args.absolute_error_thresh
            )
            results.append(
                dict(
                    junction=res.unzip[0],
                    converged=res.converged,
                    loglik=res.fit.loglik,
                    num_params=res.fit.num_params,
                    rmse=np.sqrt(np.mean(res.fit.residual**2)),
                    mae=np.mean(np.abs(res.fit.residual)),
                    completeness_ratio=res.completeness_ratio,
                    excess_paths=res.excess_paths,
                    score_margin=res.score_margin,
                    identifiable=passes_identifiability,
                    score_name=args.score_name,
                    model=str(args.depth_model),
                    paths=str(res.unzip[1]),
                )
            )

        results = pd.DataFrame(results).set_index("junction")
        results.to_csv(args.outpath, sep="\t")
