import logging

import graph_tool as gt

import strainzip as sz
from strainzip.logging_util import phase_info

from ._base import App


class ExtractResults(App):
    """Gather sequences and depths from a graph."""

    def add_custom_cli_args(self):
        self.parser.add_argument("graph_inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("fasta_inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("segments_outpath", help="Where to write segments")
        self.parser.add_argument("depth_outpath", help="Where to write depth table")
        self.parser.add_argument(
            "fasta_outpath", help="Where to write assembled sequences"
        )

    def execute(self, args):
        with phase_info("Loading data"):
            graph = sz.io.load_graph(args.graph_inpath)
            with open(args.fasta_inpath) as f:
                (
                    _,
                    unitig_to_sequence,
                ) = sz.io.load_graph_and_sequences_from_linked_fasta(
                    f, graph.gp["kmer_length"], sz.io.ggcat_header_tokenizer
                )

        with phase_info("Compiling results"):
            results = (
                sz.results.extract_vertex_data(graph)
                .assign(
                    assembly=lambda d: d.segments.apply(
                        sz.results.assemble_overlapping_unitigs,
                        unitig_to_sequence=unitig_to_sequence,
                        k=graph.gp["kmer_length"],
                    )
                )
                .sort_values(["length"], ascending=False)
            )
            depth_table = sz.results.full_depth_table(graph)

        with phase_info("Writing results"):
            with phase_info("Writing FASTA"), open(
                args.fasta_outpath, "w"
            ) as fasta_handle:
                for vertex, data in results.iterrows():
                    print(f">{vertex}\n{data.assembly}", file=fasta_handle)
            with phase_info("Writing depth"):
                depth_table.to_csv(args.depth_outpath, sep="\t")
            with phase_info("Writing segments"):
                results.segments.apply(lambda x: ",".join(x)).to_csv(
                    args.segments_outpath, sep="\t", header=False
                )
