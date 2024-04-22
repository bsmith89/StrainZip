import logging

import graph_tool as gt

import strainzip as sz

from ._base import App


class TrimTips(App):
    """Trim hanging tips."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="graph-tool formatted graph.")
        self.parser.add_argument("outpath")

    def execute(self, args):
        with sz.logging_util.phase_info("Loading graph"):
            graph = sz.io.load_graph(args.inpath)
        kmer_length = graph.gp["kmer_length"]
        logging.debug(graph)

        unzippers = [
            sz.graph_manager.LengthUnzipper(),
            sz.graph_manager.SequenceUnzipper(),
            sz.graph_manager.VectorDepthUnzipper(),
            sz.graph_manager.PositionUnzipper(offset=(0.1, 0.1)),
        ]
        pressers = [
            sz.graph_manager.LengthPresser(),
            sz.graph_manager.SequencePresser(sep=","),
            sz.graph_manager.VectorDepthPresser(),
            sz.graph_manager.PositionPresser(),
        ]

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
            tips = sz.topology.find_tips(
                graph, also_required=graph.vp["length"].a < kmer_length
            )

        with sz.logging_util.phase_info("Trimming tips"):
            logging.info(f"Removing {len(tips)} tips with length < {kmer_length}.")
            gm.batch_trim(graph, tips)
            # TODO (2024-04-22): Be super confident that the vp['filter'] is the vertex filter,
            # and therefore prune=True drops the tips.
            graph = gt.Graph(graph, prune=True)
            logging.debug(graph)

        with sz.logging_util.phase_info("Writing result"):
            sz.io.dump_graph(graph, args.outpath)
