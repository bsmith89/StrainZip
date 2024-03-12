from collections import defaultdict
from typing import Any, Callable, Optional, Sequence, Tuple, TypeAlias, cast

import graph_tool as gt
import numpy as np

from . import vertex_properties
from .exceptions import InvalidCoordValueException
from .types import ChildID, ParentID, VertexID
from .vertex_properties import (
    PropertyManager,
    depth_manager,
    filter_manager,
    length_manager,
    position_manager,
    sequence_manager,
)

AnyUnzipParam: TypeAlias = Any
AnyPressParam: TypeAlias = Any


class BaseGraphManager:
    """Base class for handling graph properties through the ZipProperty paradigm.

    This class manages a graph and its associated ZipProperty instances,
    orchestrating the unzip and press operations across both the graph topology and all properties.

    Attributes:
        props (dict[str, ZipProperty]): A dictionary mapping property names to their respective ZipProperty instances.
    """

    def __init__(self, **property_managers: PropertyManager):
        self.property_managers = property_managers

    def validate_manager(self, graph):
        # Confirm all properties are already associated with the graph.
        for k in self.property_managers:
            assert k in graph.vertex_properties

    def validate_graph(self, graph):
        # Confirm all graph properties are associated with the property managers.
        for k in graph.vp:
            assert k in self.property_managers

    def unzip(
        self,
        graph,
        parent: ParentID,
        paths: Sequence[Tuple[VertexID, VertexID]],
        **params,
    ):

        # Unzip vertex.
        n = len(paths)
        num_before = graph.num_vertices(ignore_filter=True)
        num_after = num_before + n
        graph.add_vertex(n)
        children = [cast(ChildID, i) for i in range(num_before, num_after)]
        new_edge_list = []
        for (left, right), child in zip(paths, children):
            new_edge_list.append((left, child))
            new_edge_list.append((child, right))
        graph.add_edge_list(new_edge_list)

        # Manage vertex properties.
        for prop in self.property_managers:
            self.property_managers[prop].unzip(
                graph.vertex_properties[prop],
                parent,
                children,
                **params,
            )

    def press(self, graph, parents: Sequence[ParentID], **params):
        # Press vertices.
        child = cast(
            ChildID, graph.num_vertices(ignore_filter=True)
        )  # Infer new node index by size of the graph.
        graph.add_vertex()
        leftmost_parent = parents[0]
        rightmost_parent = parents[-1]
        left_list = graph.get_in_neighbors(leftmost_parent)
        right_list = graph.get_out_neighbors(rightmost_parent)
        new_edge_list = []
        for left in left_list:
            new_edge_list.append((left, child))
        for right in right_list:
            new_edge_list.append((child, right))
        graph.add_edge_list(new_edge_list)

        # Manage vertex properties.
        for prop in self.property_managers:
            self.property_managers[prop].press(
                graph.vertex_properties[prop],
                parents,
                child,
                **params,
            )


class FilterGraphManager(BaseGraphManager):
    def __init__(self, **property_managers: PropertyManager):
        super().__init__(filter=filter_manager, **property_managers)

    def unzip(
        self,
        graph,
        parent,
        paths,
        **params,
    ):
        super().unzip(graph, parent, paths, **params)

    def press(self, graph, parents, **params):
        super().press(graph, parents, **params)


class SequenceGraphManager(FilterGraphManager):
    def __init__(
        self,
        **property_managers,
    ):
        super().__init__(
            length=length_manager,
            sequence=sequence_manager,
            **property_managers,
        )

    def unzip(
        self,
        graph,
        parent,
        paths,
        **params,
    ):
        super().unzip(graph, parent, paths, **params)

    def press(self, graph, parents, **params):
        super().press(graph, parents, **params)


class DepthGraphManager(SequenceGraphManager):
    def __init__(
        self,
        depth=depth_manager,
        **property_managers,
    ):
        super().__init__(
            depth=depth,
            **property_managers,
        )

    def unzip(
        self,
        graph,
        parent,
        paths,
        **params,
    ):
        super().unzip(graph, parent, paths, **params)

    def press(self, graph, parents, lengths=None, **params):
        if lengths is None:
            lengths = graph.vertex_properties["length"].a[parents]

        super().press(graph, parents, lengths=lengths, **params)


class PositionGraphManager(DepthGraphManager):
    def __init__(
        self,
        pos_offset_scale=0.1,
        sfdp_layout_kwargs: Optional[dict[str, Any]] = None,
        **property_managers,
    ):
        super().__init__(
            xyposition=position_manager,
            **property_managers,
        )

        # Positioning specific parameters.
        # Parameter used to spread nodes on unzip.
        self.pos_offset_scale = pos_offset_scale

        # Parameters used by update_positions() to control the sfdp_layout engine.
        if sfdp_layout_kwargs is None:
            sfdp_layout_kwargs = {}
        sfdp_layout_kwargs = (
            dict(K=0.5, init_step=0.005, max_iter=1) | sfdp_layout_kwargs
        )
        self.sfdp_layout_kwargs = sfdp_layout_kwargs

    def update_positions(self, graph, **kwargs):
        sfdp_layout_kwargs = self.sfdp_layout_kwargs | kwargs
        xyposition = gt.draw.sfdp_layout(
            graph,
            pos=graph.vertex_properties["xyposition"],
            **sfdp_layout_kwargs,
        )
        if np.isnan(xyposition.get_2d_array(pos=[0, 1])).any():
            raise InvalidCoordValueException(
                "NaN value in xcoord or ycoord. Maybe your initial values had too much symmetry?"
            )
        graph.vertex_properties["xyposition"] = xyposition

    def unzip(
        self,
        graph,
        parent,
        paths,
        pos_offsets=None,
        **params,
    ):
        if pos_offsets is None:
            pos_offsets = np.linspace(
                -self.pos_offset_scale, self.pos_offset_scale, num=len(paths)
            )
            pos_offsets = np.stack([pos_offsets] * 2)

        super().unzip(graph, parent, paths, pos_offsets=pos_offsets, **params)
        self.update_positions(graph)

    def press(self, graph, parents, lengths=None, **params):
        if lengths is None:
            lengths = graph.vertex_properties["length"].a[parents]

        super().press(
            graph,
            parents,
            lengths=lengths,
            **params,
        )
        self.update_positions(graph)
