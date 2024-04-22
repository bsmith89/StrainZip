import logging

import graph_tool as gt

import strainzip as sz

from ._base import App


class SelectLocalGraph(App):
    """Select graph within a fixed radius of a focal segment."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="graph-tool formatted graph.")
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

    def validate_and_transform_args(self, args):
        args.segment_list = args.segment_list.split(",")
        return args

    def execute(self, args):
        with sz.logging_util.phase_info("Loading graph."):
            full_graph = sz.io.load_graph(args.inpath)
            kmer_length = full_graph.gp["kmer_length"]
            logging.debug(full_graph)

        with sz.logging_util.phase_info("Finding focal vertices."):
            focal_vertices = []
            for segment in args.segment_list:
                # import pdb; pdb.set_trace()
                logging.debug(segment)
                segment_vertices = list(
                    sz.results.iter_vertices_with_segment(full_graph, segment)
                )
                logging.debug(segment_vertices)
                focal_vertices.extend(segment_vertices)
            logging.info(f"Found {len(focal_vertices)} focal vertices.")

        with sz.logging_util.phase_info("Selecting radius."):
            min_dist = sz.topology.get_shortest_distance(
                full_graph, roots=focal_vertices, weights=full_graph.vp["length"]
            )
            vfilt = full_graph.new_vertex_property(
                "bool", vals=(min_dist.a < args.radius)
            )
            logging.info(f"Found {vfilt.a.sum()} vertices inside radius.")
            local_graph = gt.GraphView(full_graph, vfilt=vfilt)
            local_graph = gt.Graph(local_graph, prune=True)
            logging.debug(local_graph)

        with sz.logging_util.phase_info("Writing result."):
            sz.io.dump_graph(local_graph, args.outpath)
