import argparse
import sys

from .app.deconvolve import DeconvolveGraph
from .app.depth import EstimateUnitigDepth
from .app.describe import DescribeGraph
from .app.example import Example
from .app.extract import ExtractResults
from .app.focus import SelectLocalGraph
from .app.load import LoadGraph
from .app.smooth import SmoothDepths
from .app.trim import TrimTips

APPLICATIONS = {
    "foo": Example,
    "load": LoadGraph,
    "depth": EstimateUnitigDepth,
    "smooth": SmoothDepths,
    "trim": TrimTips,
    "focus": SelectLocalGraph,
    "deconvolve": DeconvolveGraph,
    "extract": ExtractResults,
    "describe": DescribeGraph,
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
