import logging

import graph_tool as gt

import strainzip as sz

from ._base import App

DEFAULT_NUM_KMER_LENGTHS = 1.0


class TrimTips(App):
    """Trim hanging tips."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("outpath")
        self.parser.add_argument(
            "--no-press",
            action="store_true",
            help="Do not press non-branching paths into tigs after trimming tips.",
        )
        self.parser.add_argument(
            "--num-kmer-lengths",
            type=float,
            default=DEFAULT_NUM_KMER_LENGTHS,
            help="Multiple of kmer-length length threshold for trimming.",
        )
        self.parser.add_argument(
            "--no-purge",
            action="store_true",
            help="Keep filtered vertices instead of purging them before saving output.",
        )
        self.parser.add_argument(
            "--processes",
            "-p",
            type=int,
            default=1,
            help="Number of parallel processes.",
        )

    def execute(self, args):
        gt.openmp_set_num_threads(args.processes)
        with sz.logging_util.phase_info("Loading graph"):
            graph = sz.io.load_graph(args.inpath)
        kmer_length = graph.gp["kmer_length"]
        logging.debug(graph)

        # NOTE: (2024-08-06) Conditional unzippers and pressers so that
        # a graph without "depth" vp can have its tips trimmed.
        # TODO: (2024-08-06) Refactor this portion for all CLI tools.
        unzippers = []
        pressers = []
        if "length" in graph.vp:
            unzippers.append(sz.graph_manager.LengthUnzipper())
            pressers.append(sz.graph_manager.LengthPresser())
        if "sequence" in graph.vp:
            unzippers.append(sz.graph_manager.SequenceUnzipper())
            pressers.append(sz.graph_manager.SequencePresser(sep=","))
        if "depth" in graph.vp:
            unzippers.append(sz.graph_manager.VectorDepthUnzipper())
            pressers.append(sz.graph_manager.VectorDepthPresser())
        if "xyposition" in graph.vp:
            unzippers = unzippers.append(
                sz.graph_manager.PositionUnzipper(offset=(0.1, 0.1))
            )
            pressers = pressers.append(sz.graph_manager.PositionPresser())

        gm = sz.graph_manager.GraphManager(
            unzippers=unzippers,
            pressers=pressers,
        )
        gm.validate(graph)

        with sz.logging_util.phase_info("Finding tips"):
            length_thresh = kmer_length * args.num_kmer_lengths
            logging.info(f"Selecting tips with length less than {length_thresh}.")
            tips = sz.topology.find_tips(
                graph, also_required=graph.vp["length"].a < length_thresh
            )

        with sz.logging_util.phase_info("Trimming tips"):
            logging.info(f"Removing {len(tips)} tips.")
            gm.batch_trim(graph, tips)
            # TODO (2024-04-22): Be super confident that the vp['filter'] is the vertex filter,
            # and therefore purge=True drops the tips.
            logging.debug(graph)

        if not args.no_press:
            with sz.logging_util.phase_info("Finding non-branching paths"):
                unitig_paths = list(sz.topology.iter_maximal_unitig_paths(graph))
            with sz.logging_util.phase_info("Pressing tigs"):
                new_pressed_vertices = gm.batch_press(
                    graph,
                    *[(path, {}) for path in unitig_paths],
                )
                logging.info(
                    f"Pressed non-branching paths into {len(new_pressed_vertices)} new tigs."
                )
                logging.debug(graph)

        with sz.logging_util.phase_info("Writing result"):
            sz.io.dump_graph(graph, args.outpath, purge=(not args.no_purge))
