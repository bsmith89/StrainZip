import argparse
import sys

from . import app
from .app_depth import EstimateUnitigDepth
from .app_smooth import SmoothDepths

APPLICATIONS = {
    "load_graph": app.LoadGraph,
    "depth": EstimateUnitigDepth,
    "smooth": SmoothDepths,
}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    subparsers = parser.add_subparsers(dest="cmd", metavar="CMD", required=True)
    for invocation, app_class in APPLICATIONS.items():
        app_class(subparsers, invocation)

    # Default parsing
    raw_args = parser.parse_args(sys.argv[1:])
    raw_args.subcommand_main(raw_args)


if __name__ == "__main__":
    main()
