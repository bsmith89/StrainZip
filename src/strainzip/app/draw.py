import logging

import graph_tool as gt
import matplotlib as mpl
import numpy as np

import strainzip as sz
from strainzip.logging_util import phase_info

from ._base import App


class DrawGraph(App):
    """Plot a graph."""

    def add_custom_cli_args(self):
        self.parser.add_argument("inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("outpath", help="Where to write assembled sequences")
        self.parser.add_argument("--random-seed", default=0)
        self.parser.add_argument("--repulsive-force-exponent", default=1.4)
        self.parser.add_argument("--optimal-edge-length", default=1.0)

    def execute(self, args):
        with phase_info("Loading graph"):
            graph = sz.io.load_graph(args.inpath)

        logging.info(
            f"Graph has {graph.num_vertices()} vertices and {graph.num_edges()} edges."
        )

        gt.seed_rng(args.random_seed)
        np.random.seed(args.random_seed)
        sz.draw.update_xypositions(
            graph, layout=gt.draw.random_layout, shape=(10, 10)
        )  # NOTE: This allows for reproducible layouts.
        sz.draw.update_xypositions(
            graph,
            layout=gt.draw.sfdp_layout,
            p=args.repulsive_force_exponent,
            K=args.optimal_edge_length,
        )
        logging.debug("Done with layout.")

        vertex_color = graph.new_vertex_property(
            "float", vals=np.log(sz.results.total_depth_property(graph).a + 1)
        )
        vertex_size = graph.new_vertex_property(
            "float",
            vals=np.log(graph.vp["length"].a + 1) * 4,
        )

        sz.draw.draw_graph(
            graph,
            vertex_size=vertex_size,
            vertex_font_size=16,
            vertex_fill_color=vertex_color,
            output=args.outpath,
            vcmap=(mpl.cm.magma, 1),
            output_size=(1000, 1000),
        )
