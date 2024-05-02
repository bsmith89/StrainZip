import graph_tool as gt
import xarray as xr

import strainzip as sz

from ._base import App


class LoadGraph(App):
    """Load GGCAT to StrainZip graph file."""

    def add_custom_cli_args(self):
        self.parser.add_argument("k", type=int, help="Kmer length")
        self.parser.add_argument("fasta_inpath", help="FASTA from GGCAT")
        self.parser.add_argument("depth_inpath", help="Preloaded NetCDF depth table")
        self.parser.add_argument("outpath")
        self.parser.add_argument(
            "--drop-orphans",
            action="store_true",
            help="Drop unitigs with no neighbors.",
        )

    def execute(self, args):
        # Load topology.
        with open(args.fasta_inpath) as f:
            graph, _ = sz.io.load_graph_and_sequences_from_linked_fasta(
                f,
                k=args.k,
                header_tokenizer=sz.io.ggcat_header_tokenizer,
            )

        # Load depth onto graph.
        depth_table = xr.load_dataarray(args.depth_inpath)
        vertex_unitig_order = [int(s[:-1]) for s in graph.vp["sequence"]]
        graph.vp["depth"] = graph.new_vertex_property("vector<float>")
        graph.vp["depth"].set_2d_array(
            depth_table.sel(unitig=vertex_unitig_order).T.values
        )

        # Load metadata.
        graph.gp["kmer_length"] = graph.new_graph_property("int", val=args.k)
        graph.gp["num_samples"] = graph.new_graph_property(
            "int", val=len(depth_table.sample)
        )

        if args.drop_orphans:
            in_degree = graph.degree_property_map("in")
            out_degree = graph.degree_property_map("out")
            not_orphan = graph.new_vertex_property(
                "bool", vals=(in_degree.a + out_degree.a) > 0
            )
            graph.set_vertex_filter(not_orphan)

        # Write out.
        sz.io.dump_graph(graph, args.outpath, prune=True)
