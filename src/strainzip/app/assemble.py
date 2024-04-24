import logging
from multiprocessing import Pool as processPool
from multiprocessing.dummy import Pool as threadPool

import graph_tool as gt
import numpy as np
from tqdm import tqdm

import strainzip as sz
from strainzip.logging_util import phase_info

from .. import depth_model
from ._base import App


def _estimate_flow(args):
    graph, depth, weight = args
    # NOTE (2024-04-23): Something is wrong here when _estimate_flow is called
    # within a multiprocessing.Pool.
    flow = sz.flow.estimate_flow(
        graph, depth, weight, eps=0.001, maxiter=200, verbose=False, flow_init=None
    )[0]
    return flow


def _estimate_all_flows(graph, processes=1, verbose=False):
    if processes > 1:
        Pool = processPool  # TODO(2024-04-23): Figure out why multiprocessing.Pool doesn't work here.
    else:
        Pool = threadPool

    with Pool(processes=processes) as pool:
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
        flow = [graph.own_property(f) for f in tqdm(flow, disable=(not verbose))]
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


# TODO (2024-04-20): Move these functions into the assembly app
# instead of the assembly module.
def _calculate_junction_deconvolution(args):
    (
        junction,
        in_neighbors,
        in_flows,
        out_neighbors,
        out_flows,
        forward_stop,
        backward_stop,
        alpha,
        score_margin_thresh,
        condition_thresh,
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
        alpha=alpha,
    )

    X = sz.deconvolution.design_paths(n, m)[0]

    if not (score_margin > score_margin_thresh):
        # print(f"[junc={j} / {n}x{m}] Cannot pick best model. (Selected model had {len(paths)} paths; score margin: {score_margin})")
        pass
    elif not X[:, paths].sum(1).min() == 1:
        # print(f"[junc={j} / {n}x{m}] Non-complete. (Best model had {len(paths)} paths; score margin: {score_margin})")
        pass
    elif not len(paths) <= max(n, m):
        # print(f"[junc={j} / {n}x{m}] Non-minimal. (Best model had {len(paths)} paths; score margin: {score_margin})")
        pass
    elif not (np.linalg.cond(fit.hessian_beta) < condition_thresh):
        # print(f"[junc={j} / {n}x{m}] Non-identifiable. (Best model had {len(paths)} paths; score margin: {score_margin})")
        pass
    else:
        # print(f"[junc={j} / {n}x{m}] SUCCESS! Selected {len(paths)} paths; score margin: {score_margin}")
        return junction, named_paths, {"path_depths": np.array(fit.beta.clip(0))}


def _parallel_calculate_junction_deconvolutions(
    junctions,
    graph,
    flow,
    forward_stop=0.0,
    backward_stop=0.0,
    alpha=1.0,
    score_margin_thresh=20.0,
    condition_thresh=1e5,
    max_paths=20,
    processes=1,
):
    # FIXME (2024-04-21): This architecture means that all of the JAX
    # stuff needs to be recompiled every time in every process.
    # If I'm lucky persistent compilation cache will solve my problems
    # some day: https://github.com/google/jax/discussions/13736
    if processes > 1:
        Pool = processPool
    else:
        Pool = threadPool

    with Pool(processes=processes) as pool:
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
                    alpha,
                    score_margin_thresh,
                    condition_thresh,
                )
                for junction, in_neighbors, in_flows, out_neighbors, out_flows in _iter_junction_deconvolution_data(
                    junctions, graph, flow, max_paths=max_paths
                )
            ),
        )

        batch = []
        for result in tqdm(
            deconv_results,
            disable=(not logging.getLogger().isEnabledFor(logging.INFO)),
            total=len(junctions),
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
            default=100,
            help="Maximum rounds of graph deconvolution.",
        )
        self.parser.add_argument(
            "--condition-thresh",
            "-c",
            type=float,
            default=1e5,
            help=(
                "Maximum condition number of the Fisher Information Matrix to still deconvolve a junction. "
                " This is used as a proxy for how identifiable the depth estimates are."
            ),
        )
        self.parser.add_argument(
            "--alpha",
            "-a",
            type=float,
            default=1.0,
            help="Value of the alpha parameter to the depth model.",
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

        with phase_info("Main loop"):
            for i in range(args.max_iter):
                with phase_info(f"Round {i + 1}"):
                    with phase_info("Optimize flow"):
                        flow = _estimate_all_flows(
                            graph,
                            processes=1,  # TODO (2024-04-23): Figure out why multiprocessing.Pool doesn't work here.
                            verbose=args.debug,
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
                            forward_stop=0.0,
                            backward_stop=0.0,
                            alpha=1.0,
                            score_margin_thresh=args.score_thresh,
                            condition_thresh=args.condition_thresh,
                            max_paths=100,  # FIXME: Consider whether I want this parameter at all.
                            processes=args.processes,
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
