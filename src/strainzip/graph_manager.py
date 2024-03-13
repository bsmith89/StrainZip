from collections import defaultdict
from typing import Any, Callable, Optional, Sequence, Tuple, TypeAlias, cast

import graph_tool as gt
import numpy as np

from . import property_manager
from .exceptions import InvalidCoordValueException
from .property_manager import (
    BasePropertyManager,
    depth_manager,
    filter_manager,
    length_manager,
    position_manager,
    sequence_manager,
)
from .types import ChildID, ParentID, VertexID

AnyUnzipParam: TypeAlias = Any
AnyPressParam: TypeAlias = Any


class BaseGraphManager:
    def __init__(self, **kwargs: BasePropertyManager):
        # TODO: Check that all kwargs are actually property managers.
        self.property_managers = kwargs

    def validate_graph(self, graph):
        # Confirm all properties are already associated with the graph.
        for k in self.property_managers:
            assert k in graph.vertex_properties

    def validate_manager(self, graph):
        # Confirm all graph properties are associated with the property managers.
        for k in graph.vp:
            assert k in self.property_managers

    def _unzip_topology(self, graph, parent, paths):
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
        return children

    def _unzip_properties(
        self,
        graph,
        parent,
        children,
        **params,
    ):
        for prop in self.property_managers:
            self.property_managers[prop].unzip(
                graph.vertex_properties[prop],
                parent,
                children,
                **params,
            )

    def unzip(
        self,
        graph,
        parent: ParentID,
        paths: Sequence[Tuple[VertexID, VertexID]],
        **params,
    ):
        children = self._unzip_topology(graph, parent, paths)
        self._unzip_properties(graph, parent, children, **params)

    def _press_topology(self, graph, parents):
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

        return child

    def _press_properties(self, graph, parents, child, **params):
        # Manage vertex properties.
        for prop in self.property_managers:
            self.property_managers[prop].press(
                graph.vertex_properties[prop],
                parents,
                child,
                **params,
            )

    def press(self, graph, parents: Sequence[ParentID], **params):
        child = self._press_topology(graph, parents)
        self._press_properties(graph, parents, child, **params)

    def batch_unzip(self, graph, *args):
        for parent, paths, params in args:
            self.unzip(graph, parent, paths, **params)

    def batch_press(self, graph, *args):
        for parents, params in args:
            self.press(graph, parents, **params)


class FilterGraphManager(BaseGraphManager):
    def __init__(self, **kwargs):
        super().__init__(filter=filter_manager, **kwargs)

    def unzip(
        self,
        graph,
        parent,
        paths,
        num_children=None,
        **params,
    ):
        if num_children is None:
            num_children = len(paths)
        super().unzip(graph, parent, paths, num_children=num_children, **params)

    def press(self, graph, parents, **params):
        super().press(graph, parents, **params)


class SequenceGraphManager(FilterGraphManager):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            length=length_manager,
            sequence=sequence_manager,
            **kwargs,
        )

    def unzip(
        self,
        graph,
        parent,
        paths,
        num_children=None,
        **params,
    ):
        if num_children is None:
            num_children = len(paths)
        super().unzip(graph, parent, paths, num_children=num_children, **params)

    def press(self, graph, parents, **params):
        super().press(graph, parents, **params)


class DepthGraphManager(SequenceGraphManager):
    def __init__(
        self,
        depth=depth_manager,
        **kwargs,
    ):
        super().__init__(
            depth=depth,
            **kwargs,
        )

    def unzip(  # type: ignore[reportIncompatibleMethodOverride]
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


class VizGraphManager(DepthGraphManager):
    def __init__(
        self,
        pos_offset_scale=0.1,
        sfdp_layout_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            xyposition=position_manager,
            **kwargs,
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

    def _unzip(self, graph, parent, paths, pos_offsets=None, **params):
        if pos_offsets is None:
            pos_offsets = np.linspace(
                -self.pos_offset_scale, self.pos_offset_scale, num=len(paths)
            )
            pos_offsets = np.stack([pos_offsets] * 2)

        super().unzip(graph, parent, paths, pos_offsets=pos_offsets, **params)

    def unzip(
        self,
        graph,
        parent,
        paths,
        pos_offsets=None,
        **params,
    ):
        self._unzip(graph, parent, paths, pos_offsets=pos_offsets, **params)
        self.update_positions(graph)

    def _press(self, graph, parents, lengths=None, **params):
        if lengths is None:
            lengths = graph.vertex_properties["length"].a[parents]
        super().press(
            graph,
            parents,
            lengths=lengths,
            **params,
        )

    def press(self, graph, parents, lengths=None, **params):
        self._press(graph, parents, lengths=lengths, **params)
        self.update_positions(graph)

    def batch_unzip(self, graph, *args):
        for parent, paths, params in args:
            self._unzip(graph, parent, paths, **params)
        self.update_positions(graph)

    def batch_press(self, graph, *args):
        for parents, params in args:
            self._press(graph, parents, **params)
        self.update_positions(graph)
