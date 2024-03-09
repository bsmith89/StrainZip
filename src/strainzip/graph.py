from typing import (
    Any,
    Generic,
    NewType,
    Optional,
    Self,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
)

import graph_tool as gt
import numpy as np
from graph_tool import VertexPropertyMap

from .vertex_properties import (
    DepthProperty,
    LengthProperty,
    SequenceProperty,
    ZipProperty,
)


class ZipGraph:
    """TODO

    Handles the unzipping/pressing for the graph itself, while coordinating
    the unzipping/pressing as implemented (potentially differently) for each
    ZipProperty.
    """

    def __init__(
        self: Self,
        graph: gt.Graph,
        **props: ZipProperty,
    ):
        self.graph = graph
        self.props = props

    def unzip(
        self,
        ParentID,
    ):
        # TODO: Add (one or more) new nodes, for each linking one in-edge and one out-edge.
        # TODO: Update all of the vertex properties using their own unzip methods.
        # TODO: Filter out the old node
        pass

    def press(self, *args):
        # TODO: Add a single new node, connecting the a single in-edge and single out-edge.
        # TODO: Update all of the vertex properties using their own press methods.
        # TODO: Filter out the old nodes
        pass

    def batch_unzip(self, *args):
        # TODO: Implement a multi-unzip method
        pass

    def batch_press(self, *args):
        # TODO: Implement a multi-press method
        pass


class DepthGraph(ZipGraph):
    """Wraps a filtered graph with embedded depth, weight, and sequence information.

    Handles the unzipping/pressing for the graph itself, while coordinating
    the unzipping/pressing as implemented (potentially differently) for each
    ZipProperty.
    """

    def __init__(
        self: Self,
        graph: gt.Graph,
        depth: DepthProperty,
        length: LengthProperty,
        sequence: SequenceProperty,
        **kwargs: ZipProperty,
    ):
        super().__init__(graph, depth=depth, length=length, sequence=sequence, **kwargs)
