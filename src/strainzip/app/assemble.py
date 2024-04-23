import logging

import graph_tool as gt

import strainzip as sz
from strainzip.logging_util import phase_info

from ._base import App


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
                        flow = sz.flow.estimate_all_flows(graph)
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
                        deconvolutions = sz.deconvolution.parallel_calculate_junction_deconvolutions(
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
