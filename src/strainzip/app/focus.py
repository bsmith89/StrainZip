import logging

import graph_tool as gt

import strainzip as sz

from ._base import App


class SelectLocalGraph(App):
    """Select graph within a fixed radius of a focal segment."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument(
            "segment_list",
            help="Comma separated list. Select all vertices that include this segment in their sequence.",
        )
        self.parser.add_argument(
            "radius",
            type=int,
            help="Include other vertices within radius length of the far end of these vertices.",
        )
        self.parser.add_argument("outpath")
        self.parser.add_argument(
            "--no-prune",
            action="store_true",
            help="Keep filtered vertices instead of pruning them.",
        )

    def validate_and_transform_args(self, args):
        args.segment_list = args.segment_list.split(",")
        return args

    def execute(self, args):
        with sz.logging_util.phase_info("Loading graph."):
            graph = sz.io.load_graph(args.inpath)
            logging.debug(graph)

        with sz.logging_util.phase_info("Finding focal vertices."):
            focal_vertices = list(
                sz.results.iter_find_vertices_with_any_segment(graph, args.segment_list)
            )
            logging.info(f"Found {len(focal_vertices)} focal vertices.")

        with sz.logging_util.phase_info("Selecting radius."):
            min_dist = sz.topology.get_shortest_distance(
                graph, roots=focal_vertices, weights=graph.vp["length"]
            )
            vfilt = graph.new_vertex_property("bool", vals=(min_dist.a < args.radius))
            logging.info(f"Found {vfilt.a.sum()} vertices inside radius.")
            # TODO (2024-04-23): Confirm that already filtered vertices are still filtered.
            graph = gt.GraphView(graph, vfilt=vfilt)
            logging.debug(graph)

        with sz.logging_util.phase_info("Writing result."):
            sz.io.dump_graph(graph, args.outpath, prune=(not args.no_prune))
