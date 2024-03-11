from typing import Self, Sequence, Tuple, TypeVar, cast

import graph_tool as gt
from graph_tool import VertexPropertyMap

from .vertex_properties import (
    ChildID,
    DepthProperty,
    FilterProperty,
    LengthProperty,
    ParentID,
    SequenceProperty,
    VertexID,
    ZipProperty,
)

PropValueT = TypeVar("PropValueT")
UnzipParamT = TypeVar("UnzipParamT")
PressParamT = TypeVar("PressParamT")


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
        parent: ParentID,
        paths: Sequence[Tuple[VertexID, VertexID]],
        params,
    ):
        n = len(paths)
        num_before = len(self.graph)
        num_after = num_before + n
        self.graph.add_vertex(n)
        children = [cast(ChildID, i) for i in range(num_before, num_after)]
        new_edge_list = []
        for (left, right), child in zip(paths, children):
            new_edge_list.append((left, child))
            new_edge_list.append((child, right))
        self.graph.add_edge_list(new_edge_list)
        for prop in self.props:
            self.props[prop].unzip(parent, children, params[prop])

    def press(self, parents: Sequence[ParentID], params):
        child = cast(
            ChildID, len(self.graph)
        )  # Infer new node index by size of the graph.
        self.graph.add_vertex()
        leftmost_parent = parents[0]
        rightmost_parent = parents[-1]
        left_list = self.graph.get_in_neighbors(leftmost_parent)
        right_list = self.graph.get_out_neighbors(rightmost_parent)
        new_edge_list = []
        for left in left_list:
            new_edge_list.append((left, child))
        for right in right_list:
            new_edge_list.append((child, right))

        for prop in self.props:
            self.props[prop].press(parents, child, params[prop])

    #
    # def batch_unzip(self, *args):
    #     # TODO: Implement a multi-unzip method
    #     pass
    #
    # def batch_press(self, *args):
    #     # TODO: Implement a multi-press method
    #     pass


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

        # Automatic "filter" property.
        self.props["filter"] = FilterProperty(
            self.graph.new_vertex_property("bool", val=True)
        )
