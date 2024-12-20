import xarray as xr

import strainzip as sz
from strainzip.logging_util import phase_info, tqdm_debug

from ._base import App


class DumpResults(App):
    """Gather sequences and depths from a graph."""

    def add_custom_cli_args(self):
        self.parser.add_argument("graph_inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("fasta_inpath", help="GGCAT FASTA.")
        self.parser.add_argument("segments_outpath", help="Where to write segments")
        self.parser.add_argument("depth_outpath", help="Where to write depth table")
        self.parser.add_argument(
            "fasta_outpath", help="Where to write assembled sequences"
        )

    def execute(self, args):
        with phase_info("Loading graph"):
            graph = sz.io.load_graph(args.graph_inpath)
        with phase_info("Loading FASTA"):
            with open(args.fasta_inpath) as f:
                (
                    _,
                    unitig_to_sequence,
                ) = sz.io.load_graph_and_sequences_from_linked_fasta(
                    f, graph.gp["kmer_length"], sz.io.ggcat_header_tokenizer
                )

        with phase_info("Compiling results"):
            results = sz.results.extract_vertex_data(graph).sort_values(
                ["length"], ascending=False
            )
            # TODO (2024-11-19): Without dereplication as below?
            dereplicated_vertices = sz.results.dereplicate_vertices_by_segments(results)
            depth_table = (
                sz.results.full_depth_table(graph)
                .rename_axis(index="sequence", columns="sample")
                .rename(str)
                .T.stack()
                .to_xarray()
            )

        with phase_info("Writing results"):
            with phase_info("Writing FASTA"), open(
                args.fasta_outpath, "w"
            ) as fasta_handle:
                for segments, vertex_list in tqdm_debug(
                    dereplicated_vertices.items(), total=len(dereplicated_vertices)
                ):
                    header = "_".join([str(v) for v in vertex_list])
                    assembly = sz.results.assemble_overlapping_unitigs(
                        segments,
                        unitig_to_sequence=unitig_to_sequence,
                        k=graph.gp["kmer_length"],
                    )
                    print(f">{header}\n{assembly}", file=fasta_handle)
            with phase_info("Writing depth"):
                depth_table.to_netcdf(args.depth_outpath)
            with phase_info("Writing segments"):
                results.segments.apply(lambda x: ",".join(x)).to_csv(
                    args.segments_outpath, sep="\t", header=False
                )


class DumpContigs(App):
    """Gather assembled sequences from a graph."""

    def add_custom_cli_args(self):
        self.parser.add_argument("graph_inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("fasta_inpath", help="GGCAT FASTA.")
        self.parser.add_argument(
            "--min-depth",
            "-d",
            type=float,
            default=0,
            help="Minimum (summed) depth to output a contig.",
        )
        self.parser.add_argument(
            "--no-derep",
            action="store_true",
            help="Write out all tigs, including exact matches.",
        )
        self.parser.add_argument(
            "fasta_outpath", help="Where to write assembled sequences"
        )

    def execute(self, args):
        with phase_info("Loading graph"):
            graph = sz.io.load_graph(args.graph_inpath)
        with phase_info("Loading FASTA"):
            with open(args.fasta_inpath) as f:
                (
                    _,
                    unitig_to_sequence,
                ) = sz.io.load_graph_and_sequences_from_linked_fasta(
                    f, graph.gp["kmer_length"], sz.io.ggcat_header_tokenizer
                )

        with phase_info("Compiling results"):
            results = sz.results.extract_vertex_data(graph).sort_values(
                ["length"], ascending=False
            )
            results = results[lambda x: x.total_depth > args.min_depth]
            if args.no_derep:
                vertices = (
                    results["segments"]
                    .rename_axis("vertex")  # type: ignore[reportAttributeAccessIssue]
                    .reset_index()
                    .assign(vertex=lambda x: x.vertex.apply(list))
                    .set_index("segments")
                )
            else:
                vertices = sz.results.dereplicate_vertices_by_segments(results)

        with phase_info("Writing results"):
            with phase_info("Writing FASTA"), open(
                args.fasta_outpath, "w"
            ) as fasta_handle:
                for segments, vertex_list in tqdm_debug(
                    vertices.items(), total=len(vertices)
                ):
                    header = "_".join([str(v) for v in vertex_list])
                    assembly = sz.results.assemble_overlapping_unitigs(
                        segments,
                        unitig_to_sequence=unitig_to_sequence,
                        k=graph.gp["kmer_length"],
                    )
                    print(f">{header}\n{assembly}", file=fasta_handle)


class DumpSegments(App):
    """Gather details about unitigs from a graph."""

    def add_custom_cli_args(self):
        self.parser.add_argument("graph_inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("segments_outpath", help="Where to write segments")

    def execute(self, args):
        with phase_info("Loading graph"):
            graph = sz.io.load_graph(args.graph_inpath)
        with phase_info("Compiling results"):
            results = sz.results.extract_vertex_data(graph).sort_values(
                ["length"], ascending=False
            )
        with phase_info("Writing results"):
            with phase_info("Writing segments"):
                results.segments.apply(lambda x: ",".join(x)).to_csv(
                    args.segments_outpath, sep="\t", header=False
                )
