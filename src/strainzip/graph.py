from typing import Any, Mapping, Optional, Self, Sequence, Tuple, TypeAlias, cast

import graph_tool as gt
import graph_tool.draw as gtdraw
import numpy as np

from .exceptions import InvalidCoordValueException
from .vertex_properties import (
    ChildID,
    DepthProperty,
    FilterProperty,
    LengthProperty,
    ParentID,
    PositionProperty,
    SequenceProperty,
    VertexID,
    ZipProperty,
)

AnyUnzipParam: TypeAlias = Any
AnyPressParam: TypeAlias = Any


class BaseZipGraph:
    """TODO

    Handles the unzipping/pressing for the graph itself, while coordinating
    the unzipping/pressing as implemented (potentially differently) for each
    ZipProperty.
    """

    def __init__(
        self: Self,
        graph: gt.Graph,
        **extra_props: ZipProperty,
    ):
        self.graph = graph
        self.props = extra_props

    def unzip(
        self,
        parent: ParentID,
        paths: Sequence[Tuple[VertexID, VertexID]],
        **extra_params: AnyUnzipParam,
    ):
        n = len(paths)
        num_before = self.graph.num_vertices(ignore_filter=True)
        num_after = num_before + n
        self.graph.add_vertex(n)
        children = [cast(ChildID, i) for i in range(num_before, num_after)]
        new_edge_list = []
        for (left, right), child in zip(paths, children):
            new_edge_list.append((left, child))
            new_edge_list.append((child, right))
        self.graph.add_edge_list(new_edge_list)
        for prop in self.props:
            self.props[prop].unzip(parent, children, extra_params[prop])

    def press(self, parents: Sequence[ParentID], **extra_params: AnyPressParam):
        child = cast(
            ChildID, self.graph.num_vertices(ignore_filter=True)
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
        self.graph.add_edge_list(new_edge_list)

        for prop in self.props:
            self.props[prop].press(parents, child, extra_params[prop])

    #
    # def batch_unzip(self, *args):
    #     # TODO: Implement a multi-unzip method
    #     pass
    #
    # def batch_press(self, *args):
    #     # TODO: Implement a multi-press method
    #     pass


class SequenceZipGraph(BaseZipGraph):
    """Wraps a filtered graph with embedded length, and sequence information and and automatic filter."""

    def __init__(
        self: Self,
        graph: gt.Graph,
        length: LengthProperty,
        sequence: SequenceProperty,
        **extra_props: ZipProperty,
    ):
        super().__init__(graph, length=length, sequence=sequence, **extra_props)

        # Automatic "filter" property.
        self.props["filter"] = FilterProperty(
            self.graph.new_vertex_property("bool", val=True)
        )
        self.graph.set_vertex_filter(self.props["filter"].vprop)

    def unzip(
        self,
        parent,
        paths,
        **extra_params,
    ):
        params = dict(length=None, sequence=None, filter=None) | extra_params
        super().unzip(parent, paths, **params)

    def press(self, parents, **extra_params):
        params = dict(length=None, sequence=None, filter=None) | extra_params
        super().press(parents, **params)


class DepthZipGraph(SequenceZipGraph):
    def __init__(
        self: Self,
        graph: gt.Graph,
        length: LengthProperty,
        sequence: SequenceProperty,
        depth: DepthProperty,
        **extra_props: ZipProperty,
    ):
        super().__init__(
            graph, length=length, sequence=sequence, depth=depth, **extra_props
        )

    def unzip(
        self,
        parent,
        paths,
        **extra_params,
    ):
        params = dict() | extra_params
        super().unzip(parent, paths, **params)

    def press(self, parents, **extra_params):
        params = dict(depth=self.props["length"].vprop.a[parents]) | extra_params
        super().press(parents, **params)


class VizZipGraph(DepthZipGraph):
    def __init__(
        self: Self,
        graph: gt.Graph,
        depth: DepthProperty,
        length: LengthProperty,
        sequence: SequenceProperty,
        xyposition: PositionProperty,
        pos_offset_scale=0.1,
        sfdp_layout_kwargs: Optional[dict[str, Any]] = None,
        **extra_props: ZipProperty,
    ):

        if sfdp_layout_kwargs is None:
            sfdp_layout_kwargs = {}
        sfdp_layout_kwargs = (
            dict(K=0.5, init_step=0.005, max_iter=1) | sfdp_layout_kwargs
        )
        self.sfdp_layout_kwargs = sfdp_layout_kwargs

        self.pos_offset_scale = pos_offset_scale

        super().__init__(
            graph,
            depth=depth,
            length=length,
            sequence=sequence,
            xyposition=xyposition,
            **extra_props,
        )
        self.update_coords()

    def update_coords(self):
        xyposition = gtdraw.sfdp_layout(
            self.graph,
            pos=self.props["xyposition"].vprop,
            **self.sfdp_layout_kwargs,
        )
        if np.isnan(xyposition.get_2d_array(pos=[0, 1])).any():
            raise InvalidCoordValueException(
                "NaN value in xcoord or ycoord. Maybe your initial values had a symmetry?"
            )

        self.props["xyposition"] = PositionProperty(xyposition)

    def unzip(
        self,
        parent,
        paths,
        **extra_params,
    ):
        coord_offsets = np.linspace(
            -self.pos_offset_scale, self.pos_offset_scale, num=len(paths)
        )
        coord_offsets = np.stack([coord_offsets] * 2)

        params = dict(xyposition=coord_offsets) | extra_params
        super().unzip(parent, paths, **params)
        self.update_coords()

    def press(self, parents, **extra_params):
        params = (
            dict(
                xyposition=self.props["length"].vprop.a[parents],
            )
            | extra_params
        )
        super().press(parents, **params)
        self.update_coords()
