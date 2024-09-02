import os
import sqlite3
import sys
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

import strainzip as sz

from ._base import App


def _unitig_depth(arg):
    db_uri, k, n, unitig_id, sequence = arg
    con = sqlite3.connect(db_uri, uri=True)
    mean_depth = sz.io.load_mean_sequence_depth(con, sequence, k, n)
    return unitig_id, len(sequence) - k + 1, mean_depth


def _iter_unitigs(fasta_iter):
    for header, sequence in fasta_iter:
        unitig_id_string, *_ = sz.io.generic_header_tokenizer(header)
        unitig_id = unitig_id_string[1:]
        yield int(unitig_id), sequence


class EstimateUnitigDepth(App):
    """Estimate mean kmer depth of sequences."""

    def add_custom_cli_args(self):
        self.parser.add_argument(
            "fasta_inpath", help="FASTA of sequences to be quantified"
        )
        self.parser.add_argument("counts_inpath", help="SQLite3 DB of kmer counts")
        self.parser.add_argument("k", type=int, help="Kmer length")
        self.parser.add_argument("sample_list", help="Comma-separated list of samples.")
        self.parser.add_argument("outpath")
        self.parser.add_argument(
            "--preload",
            action="store_true",
            help="Pre-load the kmer counts DB into memory.",
        )
        # TODO (2024-04-23): Combine preload and tmpdb into one flag using nargs="?"
        self.parser.add_argument(
            "--tmpdb",
            help=(
                "If --preload is given, load the data into this file. "
                "If not provided, an (explicitly) in-memory DB is used instead. "
                "Has no effect if not preloading the DB. "
                "To keep memory speeds without allocating in every process, "
                "consider preloading to /dev/shm or equivalent."
            ),
        )
        self.parser.add_argument(
            "--processes",
            "-p",
            type=int,
            default=1,
            help="Number of parallel processes.",
        )

    def validate_and_transform_args(self, args):
        args.sample_list = args.sample_list.split(",")
        return args

    def execute(self, args):
        if args.preload:
            # FIXME: Switch to logging instead of print statements.
            print("Preloading counts DB.", file=sys.stderr)
            assert os.path.exists(args.counts_inpath)
            disk_con = sqlite3.connect(args.counts_inpath)
            if not args.tmpdb:
                db_uri = "file:tmpdb?mode=memory&cache=shared"
            else:
                db_uri = f"file:{args.tmpdb}?cache=shared"
            inmem_con = sqlite3.connect(db_uri, uri=True)
            disk_con.backup(inmem_con)
            disk_con.close()
            print("Finished.", file=sys.stderr)
        else:
            db_uri = f"file:{args.counts_inpath}?cache=shared"

        print(f"Working with DB at {db_uri} .")

        print("Start calculating depths.")
        results = {}
        pool = Pool(processes=args.processes)
        with open(args.fasta_inpath) as f, tqdm(mininterval=1, unit_scale=True) as pbar:
            unitig_iter = _iter_unitigs(sz.io.iter_linked_fasta_entries(f))
            results_iter = pool.imap_unordered(
                _unitig_depth,
                (
                    # FIXME (2024-05-08): Pass an explicity list of samples (instead of just the count)
                    # so that we can count depths for each one individually.
                    (db_uri, args.k, len(args.sample_list), unitig_id, sequence)
                    for unitig_id, sequence in unitig_iter
                ),
                chunksize=1000,
            )
            for unitig_id, num_kmers, depths_mean in results_iter:
                results[int(unitig_id)] = depths_mean
                pbar.update(num_kmers)
        results = pd.DataFrame(results.values(), index=results.keys(), columns=args.sample_list)  # type: ignore[reportArgumentType]
        results = (
            results.rename_axis(index="unitig", columns="sample").stack().to_xarray()
        )
        results.to_netcdf(args.outpath)
