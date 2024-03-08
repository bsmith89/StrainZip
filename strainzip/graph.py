import graph_tool as gt
import pandas as pd
from .io import load_kmer_depths
from .io import parse_gfa
from collections.abc import Iterable
from typing import TypeAlias, Self
from warnings import warn


VertexID: TypeAlias = int
Unitig: TypeAlias = list[VertexID]
Count: TypeAlias = int

class Junction:
    def __init__(self) -> None:
        # TODO: Implement some sort of data type that
        # tracks all the in- and out- edges, the VertexID, and the depth information
        # This will then be used to construct the optimization problem before
        # returning a collection of LocalPaths
        pass


class LocalPath:
    def __init__(self: Self) -> None:
        # an in-edge, out-edge, and depth
        pass


class DepthGraph:
    """Wraps a graph with embedded depth and weight information."""

    def __init__(self: Self, graph: gt.Graph, num_samples: Count) -> None:
        self.num_samples = num_samples  # Keep track of the number of samples.
        self.graph = graph  # The core graph itself.
        # TODO: Need to load this graph from somewhere and check that it already has vp['depths'] and vp['weights'].
        self.edge_weights_up_to_date = False

    def refresh_edge_weights(self: Self) -> None:
        if self.edge_weights_up_to_date:
            warn(
                "Edge weights are being refreshed, but they were already marked as up-to-date."
            )
        # TODO: Run the message passing algorithm to estimate edge weights for each sample.
        self.edge_weights_up_to_date = True

    def iter_junctions(self: Self) -> Iterable[Junction]:
        # TODO: Return depth information for in-edges, out-edges, and vertex v across all n samples.
        for i, _ in self.graph.vertices:
            yield Junction()

    def split_junctions(self: Self, local_paths: List[LocalPath]) -> None:
        self.edge_weights_up_to_date = False
        # TODO: Add vertices and edges for each new local path and replace the original junction.
        pass

    def iter_unitigs(self: Self) -> Iterable[Unitig]:
        pass
