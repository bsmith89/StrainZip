import xarray as xr

import strainzip as sz
from strainzip.logging_util import phase_info

from ._base import App


class LoadGraph(App):
    """Load GGCAT to StrainZip graph file."""

    def add_custom_cli_args(self):
        self.parser.add_argument("k", type=int, help="Kmer length")
        self.parser.add_argument("fasta_inpath", help="FASTA from GGCAT")
        self.parser.add_argument(
            "--depth", dest="depth_inpath", help="Preloaded NetCDF depth table"
        )
        self.parser.add_argument("outpath")

    def execute(self, args):
        with phase_info("Loading graph"):
            with open(args.fasta_inpath) as f:
                graph, _ = sz.io.load_graph_and_sequences_from_linked_fasta(
                    f,
                    k=args.k,
                    header_tokenizer=sz.io.ggcat_header_tokenizer,
                    verbose=args.verbose,
                )

        if args.depth_inpath:
            with phase_info("Loading depth"):
                depth_table = xr.load_dataarray(args.depth_inpath)
                vertex_unitig_order = [int(s[:-1]) for s in graph.vp["sequence"]]
                graph.vp["depth"] = graph.new_vertex_property("vector<float>")
                graph.vp["depth"].set_2d_array(
                    depth_table.sel(unitig=vertex_unitig_order).T.values
                )
                graph.gp["num_samples"] = graph.new_graph_property(
                    "int", val=len(depth_table.sample)
                )

        with phase_info("Finalizing graph object"):
            graph.gp["kmer_length"] = graph.new_graph_property("int", val=args.k)

        with phase_info("Writing output"):
            sz.io.dump_graph(graph, args.outpath)


class AugmentWithDepth(App):
    """Add depth data to a StrainZip graph file."""

    def add_custom_cli_args(self):
        self.parser.add_argument("graph_inpath", help="StrainZip formatted graph file")
        self.parser.add_argument("depth_inpath", help="Preloaded NetCDF depth table")
        self.parser.add_argument("outpath")

    def execute(self, args):
        with phase_info("Loading graph"):
            graph = sz.io.load_graph(args.graph_inpath)

        with phase_info("Loading depth"):
            depth_table = xr.load_dataarray(args.depth_inpath)
            vertex_unitig_order = [int(s[:-1]) for s in graph.vp["sequence"]]
            graph.vp["depth"] = graph.new_vertex_property("vector<float>")
            graph.vp["depth"].set_2d_array(
                depth_table.sel(unitig=vertex_unitig_order).T.values
            )
            graph.gp["num_samples"] = graph.new_graph_property(
                "int", val=len(depth_table.sample)
            )

        with phase_info("Writing output"):
            sz.io.dump_graph(graph, args.outpath)
