import logging

import graph_tool as gt
import xarray as xr

import strainzip as sz
from strainzip.logging_util import phase_info

from ._base import App


class ExtractResults(App):
    """Gather sequences and depths from a graph."""

    def add_custom_cli_args(self):
        self.parser.add_argument("graph_inpath", help="StrainZip formatted graph.")
        self.parser.add_argument("fasta_inpath", help="GGCAT FASTA.")
        self.parser.add_argument("depth_inpath", help="Depth table (NetCDF).")
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
            unitig_depth_table = xr.load_dataarray(args.depth_inpath)
            unitig_depth_table["unitig"] = unitig_depth_table["unitig"].astype(str)

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
            with phase_info("Re-adding orphan sequences"):
                unitigs_in_main_results = set(
                    results.segments.explode().drop_duplicates()
                )
                unitigs_not_in_results = (
                    set(unitig_to_sequence) - unitigs_in_main_results
                )
                orphan_depth_table = unitig_depth_table.sel(
                    unitig=list(unitigs_not_in_results)
                )
                orphan_depth_table["unitig"] = [
                    f"orphan_{u}" for u in orphan_depth_table["unitig"].values
                ]
                assembly_depth_table = (
                    sz.results.full_depth_table(graph)
                    .rename_axis(index="sequence", columns="sample")
                    .rename(str)
                    .T.stack()
                    .to_xarray()
                )
                depth_table = xr.concat(
                    [
                        assembly_depth_table,
                        orphan_depth_table.rename({"unitig": "sequence"}),
                    ],
                    dim="sequence",
                )

        with phase_info("Writing results"):
            with phase_info("Writing FASTA"), open(
                args.fasta_outpath, "w"
            ) as fasta_handle:
                for vertex, data in results.iterrows():
                    print(f">{vertex}\n{data.assembly}", file=fasta_handle)
                with phase_info("Writing orphan unitigs"):
                    for unitig in unitig_to_sequence:
                        if unitig not in unitigs_in_main_results:
                            print(
                                f">{unitig}\n{unitig_to_sequence[unitig]}",
                                file=fasta_handle,
                            )
            with phase_info("Writing depth"):
                depth_table.to_netcdf(args.depth_outpath)
            with phase_info("Writing segments"):
                results.segments.apply(lambda x: ",".join(x)).to_csv(
                    args.segments_outpath, sep="\t", header=False
                )
