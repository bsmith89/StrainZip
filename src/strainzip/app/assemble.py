import logging
from multiprocessing import Pool

import graph_tool as gt
import numpy as np

import strainzip as sz
from strainzip.logging_util import phase_info

from ..depth_model import LogPlusAlphaLogNormal
from ..depth_model2 import SoftPlusNormal
from ..logging_util import tqdm_debug
from ._base import App

DEFAULT_MAX_ITER = 100
DEFAULT_CONDITION_THRESH = 1e5

DEPTH_MODELS = {
    "LogPlusAlphaLogNormal": (LogPlusAlphaLogNormal, dict(alpha=1.0)),
    "SoftPlusNormal": (SoftPlusNormal, dict()),
}

DEFAULT_DEPTH_MODEL = "LogPlusAlphaLogNormal"


def _estimate_flow(args):
    graph, depth, length = args
    # NOTE (2024-04-23): Something is wrong here when _estimate_flow is called
    # within a multiprocessing.Pool.
    flow = sz.flow.estimate_flow(
        graph,
        depth,
        length,
        eps=0.001,
        maxiter=200,
        flow_init=None,
        ifnotconverged="error",
    )[0]
    return flow


def _parallel_estimate_all_flows(graph, pool):
    # if processes > 1:
    #     Pool = processPool  # TODO(2024-04-23): Figure out why multiprocessing.Pool doesn't work here.
    # else:
    #     Pool = threadPool
    flow = pool.imap(
        _estimate_flow,
        (
            (
                graph,
                gt.ungroup_vector_property(graph.vp["depth"], pos=[sample_id])[0],
                graph.vp["length"],
            )
            for sample_id in range(graph.gp["num_samples"])
        ),
    )
    flow = [
        graph.own_property(f)
        for f in tqdm_debug(
            flow,
            total=graph.gp["num_samples"],
            bar_format="{l_bar}{r_bar}",
        )
    ]
    flow = gt.group_vector_property(flow, pos=range(graph.gp["num_samples"]))
    return flow


def _iter_junction_deconvolution_data(junction_iter, graph, flow, max_paths):
    for j in junction_iter:
        in_neighbors = graph.get_in_neighbors(j)
        out_neighbors = graph.get_out_neighbors(j)
        n, m = len(in_neighbors), len(out_neighbors)
        if n * m > max_paths:
            continue

        # Collect flows
        # print(in_neighbors, j, out_neighbors)
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
    fit, paths, named_paths, score_margin = sz.deconvolution.deconvolve_junction(
        in_neighbors,
        in_flows,
        out_neighbors,
        out_flows,
        model=depth_model,  # TODO (2024-04-20): Allow this to be passed in by changing it from a module into a class.
        forward_stop=forward_stop,
        backward_stop=backward_stop,
    )

    X = sz.deconvolution.design_paths(n, m)[0]

    if not (score_margin > score_margin_thresh):
        # print(f"[junc={j} / {n}x{m}] Cannot pick best model. (Selected model had {len(paths)} paths; score margin: {score_margin})")
        return None

    if not X[:, paths].sum(1).min() == 1:
        # print(f"[junc={j} / {n}x{m}] Non-complete. (Best model had {len(paths)} paths; score margin: {score_margin})")
        return None

    if not len(paths) <= max(n, m):
        # print(f"[junc={j} / {n}x{m}] Non-minimal. (Best model had {len(paths)} paths; score margin: {score_margin})")
        return None

    try:
        condition = np.linalg.cond(fit.hessian_beta)
    except np.linalg.LinAlgError:
        return None
    else:
        if not (condition < condition_thresh):
            # print(f"[junc={j} / {n}x{m}] Non-identifiable. (Best model had {len(paths)} paths; score margin: {score_margin})")
            return None

    # print(f"[junc={j} / {n}x{m}] SUCCESS! Selected {len(paths)} paths; score margin: {score_margin}")
    return junction, named_paths, {"path_depths": np.array(fit.beta.clip(0))}


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
    for result in tqdm_debug(
        deconv_results,
        total=len(junctions),
        bar_format="{l_bar}{r_bar}",
    ):
        if result is not None:
            junction, named_paths, path_depths_dict = result
            # print(f"{junction}: {named_paths}", end=" | ")
            batch.append((junction, named_paths, path_depths_dict))

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
            "--max-iter",
            "-n",
            type=int,
            default=DEFAULT_MAX_ITER,
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
            ), f"{k} does not appear to be a hyperparameters of {depth_model_class}."
            model_hyperparameters[k] = float(v)

        # Instantiate the depth model and assign it to args.
        args.depth_model = depth_model_class(
            **(model_default_hyperparameters | model_hyperparameters)
        )

        return args

    def execute(self, args):
        gt.openmp_set_num_threads(args.processes)
        if args.debug or args.verbose:
            logging.getLogger("jax").setLevel(logging.CRITICAL)

        with phase_info("Loading graph"):
            graph = sz.io.load_graph(args.inpath)
            graph.vp["touched"] = graph.new_vertex_property("bool", val=False)
            gm = sz.graph_manager.GraphManager(
                unzippers=[
                    sz.graph_manager.LengthUnzipper(),
                    sz.graph_manager.SequenceUnzipper(),
                    sz.graph_manager.VectorDepthUnzipper(),
                    sz.graph_manager.TouchedUnzipper(),
                ],
                pressers=[
                    sz.graph_manager.LengthPresser(),
                    sz.graph_manager.SequencePresser(sep=","),
                    sz.graph_manager.VectorDepthPresser(),
                    sz.graph_manager.TouchedPresser(),
                ],
            )
            gm.validate(graph)
            logging.debug(graph)

        with phase_info("Main loop"), Pool(processes=args.processes) as pool:
            logging.debug(
                f"Initialized multiprocessing pool with {args.processes} workers."
            )
            for i in range(args.max_iter):
                with phase_info(f"Round {i + 1}"):
                    with phase_info("Optimize flow"):
                        flow = _parallel_estimate_all_flows(
                            graph,
                            pool,
                            # processes=args.processes,  # FIXME (2024-05-06): Figure out why multiprocessing.Pool doesn't work here.
                        )
                    with phase_info("Finding junctions"):
                        if i == 0:
                            # For first iteration, ignore the "touched" property in finding junctions.
                            junctions = sz.topology.find_junctions(graph)
                            gt.map_property_values(
                                graph.vp["touched"],
                                graph.vp["touched"],
                                lambda _: False,
                            )
                        else:
                            assert (graph.vp["touched"].fa == 1).any()
                            # For all subsequent iterations, only gather junctions that have been touched
                            # or where their neighbor has been touched.
                            consider = sz.topology.vertex_or_neighbors(
                                graph, graph.vp["touched"]
                            )
                            logging.info(
                                f"Only considering the {graph.vp['touched'].fa.sum()} vertices "
                                "affected by the previous round of deconvolution."
                            )
                            junctions = sz.topology.find_junctions(
                                graph, also_required=consider.a
                            )
                            # After finding these junctions, reset touched property, to be updated
                            # during unzipping and pressing.
                            gt.map_property_values(
                                graph.vp["touched"],
                                graph.vp["touched"],
                                lambda _: False,
                            )
                        logging.info(f"Found {len(junctions)} junctions.")
                        assert (graph.vp["touched"].fa == 0).all()
                    with phase_info("Optimizing junction deconvolutions"):
                        deconvolutions = _parallel_calculate_junction_deconvolutions(
                            junctions,
                            graph,
                            flow,
                            args.depth_model,
                            pool=pool,
                            forward_stop=0.0,
                            backward_stop=0.0,
                            score_margin_thresh=args.score_thresh,
                            condition_thresh=args.condition_thresh,
                            max_paths=100,  # FIXME: Consider whether I want this parameter at all.
                            # processes=args.processes,
                        )
                    with phase_info("Unzipping junctions"):
                        new_unzipped_vertices = gm.batch_unzip(graph, *deconvolutions)
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
            sz.io.dump_graph(graph, args.outpath, prune=(not args.no_prune))
