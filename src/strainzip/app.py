import logging
import os
import sqlite3
import sys

import pandas as pd
from tqdm import tqdm

import strainzip as sz


class App:
    """Base-class for subcommand "applications".

    Custom subcommands must subclass App (or a subclass of App) and override
    the following methods:

    - add_custom_cli_args
    - validate_and_transform_args
    - execute

    Subcommands can then be registered in sz.__main__ by adding them to the
    *APPLICATIONS* dict.

    """

    def __init__(self, subparsers, incantation):
        self.parser = subparsers.add_parser(incantation, help=self._help)
        self._add_cli_args()
        self.parser.set_defaults(subcommand_main=self._run)

    @property
    def _help(self):
        return self.__doc__

    def _add_cli_args(self):
        # Default arguments
        self.parser.add_argument(
            "--version",
            action="version",
            version=sz.__version__,
        )
        self.parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Print info messages to stderr.",
        )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="Print debug messages to stderr.",
        )

        # App-specific setup.
        self.add_custom_cli_args()

    def _process_args(self, raw_args):
        # App specific parsing/validation/transformation
        try:
            final_args = self.validate_and_transform_args(raw_args)
        except AssertionError as err:
            logging.error(f"Argument validation failed: {err}")
            sys.exit(1)

        return final_args

    def _run(self, raw_args):
        args = self._process_args(raw_args)

        # Setup logging
        logging.basicConfig(format="%(asctime)s %(message)s")
        if args.debug:
            logging_level = logging.DEBUG
        elif args.verbose:
            logging_level = logging.INFO
        else:
            logging_level = logging.WARNING
        logging.getLogger().setLevel(logging_level)
        logging.debug(f"Logging level set to {logging_level}.")

        logging.debug(f"Arguments: {args}")

        # Run the app specific work.
        self.execute(args)

    def add_custom_cli_args(self):
        """Add app-specific CLI args to parser.

        Subclasses of App must override this method.
        Arguments may be added to self.parser, an
        instance of argparse.Parser.  # TODO: Check this.

        """
        raise NotImplementedError

    def validate_and_transform_args(self, args):
        """Add custom argument validation/transformation/parsing.

        Subclasses of App must override this method.

        This method is used by subclasses to
        implement any validation/transformation/parsing of
        the "raw" argparse.Namespace object
        that cannot be implemented easily within the argparse.Parser API.

        Subclass methods must operate only on the *args* input
        and must return an argparse.Namespace for usage during execution.

        """
        return args

    def execute(self, args):
        """Implement all remaining work for the specific application.

        Subclasses of App must override this method.

        Execution should rely only on the properties of *args*.

        """
        raise NotImplementedError


class Example(App):
    """Example application to test the API."""

    def add_custom_cli_args(self):
        self.parser.add_argument("--foo", action="store_true", help="Should I foo?")
        self.parser.add_argument("--num", type=int, default=1, help="How many times?")

    def validate_and_transform_args(self, args):
        assert args.num < 5, "NUM must be less than 5"
        return args

    def execute(self, args):
        if args.foo:
            for i in range(args.num):
                print("Foo!")
        else:
            print("Nope, that's a bar.")


class EstimateUnitigDepth(App):
    """Estimate mean kmer depth of sequences."""

    def add_custom_cli_args(self):
        self.parser.add_argument("counts_inpath", help="SQLite3 DB of kmer counts")
        self.parser.add_argument("k", type=int, help="Kmer length")
        self.parser.add_argument(
            "fasta_inpath", help="FASTA of sequences to be quantified"
        )
        self.parser.add_argument("outpath")

    def execute(self, args):
        print("Start loading counts DB.", file=sys.stderr)
        assert os.path.exists(args.counts_inpath)
        disk_con = sqlite3.connect(args.counts_inpath)
        con = sqlite3.connect(":memory:")
        disk_con.backup(con)
        disk_con.close()
        print("Finished loading counts DB.", file=sys.stderr)

        print("Start calculating depths.")
        results = {}
        with open(args.fasta_inpath) as f, tqdm(mininterval=1) as pbar:
            for header, sequence in sz.io.iter_linked_fasta_entries(f):
                unitig_id_string, *_ = sz.io.ggcat_header_tokenizer(header)
                unitig_id = unitig_id_string[1:]
                depths_matrix = sz.io.load_sequence_depth_matrix(
                    con, sequence, k=args.k
                )
                depths_mean = depths_matrix.mean(0)
                results[int(unitig_id)] = depths_mean
                pbar.update(depths_mean.shape[0])
        results = pd.DataFrame(results.values(), index=results.keys())  # type: ignore[reportArgumentType]
        results = (
            results.rename_axis(index="unitig", columns="sample").stack().to_xarray()
        )
        results.to_netcdf(args.outpath)


class LoadGraph(App):
    """Load GGCAT to StrainZip graph file."""

    def add_custom_cli_args(self):
        self.parser.add_argument("k", type=int, help="Kmer length")
        self.parser.add_argument("fasta_inpath", help="FASTA from GGCAT")
        self.parser.add_argument("outpath")

    def execute(self, args):
        with open(args.fasta_inpath) as f:
            graph, _ = sz.io.load_graph_and_sequences_from_linked_fasta(
                f, k=args.k, header_tokenizer=sz.io.ggcat_header_tokenizer
            )
        sz.io.dump_graph(graph, args.outpath)
