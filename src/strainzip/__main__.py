import argparse
import sys

from .app.cluster import ClusterTigs
from .app.depth import EstimateUnitigDepth
from .app.dump import DumpResults
from .app.example import Example
from .app.focus import SelectLocalGraph
from .app.info import ShowGraphStats
from .app.load import LoadGraph
from .app.precluster import PreClusterTigs
from .app.smooth import SmoothDepths
from .app.trim import TrimTips
from .app.unzip import BenchmarkDepthModel, UnzipGraph

APPLICATIONS = {
    # "foo": Example,
    "load": LoadGraph,
    "depth": EstimateUnitigDepth,
    "smooth": SmoothDepths,
    "trim": TrimTips,
    "focus": SelectLocalGraph,
    "unzip": UnzipGraph,
    "benchmark": BenchmarkDepthModel,
    "dump": DumpResults,
    "info": ShowGraphStats,
    "cluster": ClusterTigs,
    "precluster": PreClusterTigs,
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
