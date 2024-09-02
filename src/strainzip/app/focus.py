import logging

import graph_tool as gt
import pandas as pd

import strainzip as sz

from ._base import App


class SelectLocalGraph(App):
    """Select graph within a fixed radius of a focal segment."""

    def add_custom_cli_args(self):
        self.parser.add_argument("graph_inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("outpath")
        self.parser.add_argument(
            "--vertices",
            help="Path to a file listing vertices to fetch.",
        )
        self.parser.add_argument(
            "--segments",
            help="Path to a file listing segments to fetch. Select all vertices that include any of these segments in their sequence.",
        )
        self.parser.add_argument(
            "--radius",
            type=int,
            default=0,
            help="Include other vertices within radius length of the far end of these vertices.",
        )
        self.parser.add_argument(
            "--no-purge",
            action="store_true",
            help="Keep filtered vertices instead of purging them before saving output.",
        )

    def validate_and_transform_args(self, args):
        if (args.vertices is None) and (args.segments is None):
            raise RuntimeError("One of --vertices or --segments must be provided.")
        return args

    def execute(self, args):
        with sz.logging_util.phase_info("Loading inputs"):
            graph = sz.io.load_graph(args.graph_inpath)
            logging.debug(graph)

            focal_vertices = []
            if args.segments:
                with sz.logging_util.phase_info("Adding segments"), open(
                    args.segments
                ) as f:
                    segment_list = [line.strip() for line in f]
                    logging.info(f"Adding {len(segment_list)} segments.")
                    logging.debug(segment_list)
                focal_vertices.extend(
                    list(
                        sz.results.iter_find_vertices_with_any_segment(
                            graph, segment_list
                        )
                    )
                )
            if args.vertices:
                with sz.logging_util.phase_info("Adding vertices"), open(
                    args.vertices
                ) as f:
                    vertex_list = [int(line.strip()) for line in f]
                    logging.info(f"Adding {len(vertex_list)} vertices.")
                    logging.debug(vertex_list)
                focal_vertices.extend(vertex_list)

        with sz.logging_util.phase_info("Finding core vertices"):
            logging.info(f"Found {len(focal_vertices)} core vertices.")

        if args.radius > 0:
            with sz.logging_util.phase_info("Selecting radius"):
                min_dist = sz.topology.get_shortest_distance_to_any_vertex(
                    graph,
                    roots=focal_vertices,
                    length=graph.vp["length"],
                )
                vfilt = graph.new_vertex_property(
                    "bool", vals=(min_dist.a <= args.radius)
                )
                logging.info(
                    f"Found a total of {vfilt.a.sum()} vertices inside radius."
                )
        else:
            # TODO (2024-05-29): Confirm that get_vertices includes all vertices and/or that graph doesn't have anything filtered out.
            logging.info("Extracting only core vertices.")
            vfilt = graph.new_vertex_property(
                "bool",
                vals=pd.Index(graph.get_vertices())
                .to_series()
                .isin(focal_vertices)
                .values,
            )

        # TODO (2024-04-23): Confirm that already filtered vertices are still filtered.
        graph = gt.GraphView(graph, vfilt=vfilt)
        logging.info(graph)

        with sz.logging_util.phase_info("Writing result"):
            sz.io.dump_graph(graph, args.outpath, purge=(not args.no_purge))
