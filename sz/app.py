from spgc.estimation import partition_gene_content
import spgc
import logging
import sys
import pandas as pd


class App:
    """Base-class for subcommand "applications".

    Custom subcommands must subclass App (or a subclass of App) and override
    the following methods:

    - add_custom_cli_args
    - validate_and_transform_args
    - execute

    Subcommands can then be registered in spgc.__main__ by adding them to the
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
            version=spgc.__version__,
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
        raise NotImplementedError

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
