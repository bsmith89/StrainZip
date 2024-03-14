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
    def __init__(self):
        self.property_managers = {}
        self._register_property_managers(
            # Empty registration as a template.
            # Subclass __init__'s should call this method
            # for additional PMs.
        )

    def _register_property_managers(self, **kwargs):
        # TODO: Check that all kwargs are actually property managers.
        self.property_managers |= kwargs

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

    def _unzip_setup_params(self, graph, parent, children, num_children=None, **kwargs):
        # Subclasses extend.
        if num_children is None:
            kwargs["num_children"] = len(children)
        return kwargs

    def _unzip_properties(
        self,
        graph,
        parent,
        children,
        **kwargs,
    ):
        for prop in self.property_managers:
            self.property_managers[prop].unzip(
                graph.vertex_properties[prop],
                parent,
                children,
                **kwargs,
            )

    def _unzip_finalize(self, graph, **kwargs):
        # Subclasses override
        pass

    def unzip(
        self,
        graph,
        parent: ParentID,
        paths: Sequence[Tuple[VertexID, VertexID]],
        **kwargs,
    ):
        children = self._unzip_topology(graph, parent, paths)
        kwargs = self._unzip_setup_params(graph, parent, children, **kwargs)
        self._unzip_properties(graph, parent, children, **kwargs)
        self._unzip_finalize(graph, **kwargs)

    def batch_unzip(self, graph, *args, **kwargs):
        for parent, paths, params in args:
            children = self._unzip_topology(graph, parent, paths)
            params = self._unzip_setup_params(
                graph, parent, children, **params, **kwargs
            )
            self._unzip_properties(graph, parent, children, **params)
        self._unzip_finalize(graph, **kwargs)

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

    def _press_setup_params(self, graph, parents, child, **kwargs):
        # Subclasses extend.
        return kwargs

    def _press_properties(self, graph, parents, child, **kwargs):
        # Manage vertex properties.
        for prop in self.property_managers:
            self.property_managers[prop].press(
                graph.vertex_properties[prop],
                parents,
                child,
                **kwargs,
            )

    def _press_finalize(self, graph, **kwargs):
        # Subclasses override
        pass

    def press(self, graph, parents: Sequence[ParentID], **kwargs):
        child = self._press_topology(graph, parents)
        kwargs = self._press_setup_params(graph, parents, child, **kwargs)
        self._press_properties(graph, parents, child, **kwargs)
        self._press_finalize(graph, **kwargs)

    def batch_press(self, graph, *args, **kwargs):
        for parents, params in args:
            child = self._press_topology(graph, parents)
            params = self._press_setup_params(graph, parents, child, **params, **kwargs)
            self._press_properties(graph, parents, child, **params)
        self._press_finalize(graph, **kwargs)


class FilterGraphManager(BaseGraphManager):
    def __init__(self):
        super().__init__()
        self._register_property_managers(filter=filter_manager)

    def _unzip_setup_params(self, graph, parent, children, num_children=None, **kwargs):
        if num_children is None:
            kwargs["num_children"] = len(children)
        return super()._unzip_setup_params(graph, parent, children, **kwargs)


class SequenceGraphManager(FilterGraphManager):
    def __init__(
        self,
    ):
        super().__init__()
        self._register_property_managers(
            length=length_manager,
            sequence=sequence_manager,
        )


class DepthGraphManager(SequenceGraphManager):
    def __init__(
        self,
    ):
        super().__init__()
        self._register_property_managers(
            depth=depth_manager,
        )

    def _press_setup_params(self, graph, parents, child, lengths=None, **kwargs):
        if lengths is None:
            kwargs["lengths"] = graph.vertex_properties["length"].a[parents]
        return super()._press_setup_params(graph, parents, child, **kwargs)


class VizGraphManager(DepthGraphManager):
    def __init__(
        self,
        pos_offset_scale=0.1,
        sfdp_layout_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self._register_property_managers(
            xyposition=position_manager,
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

    def _unzip_setup_params(
        self, graph, parent, children, num_children=None, pos_offsets=None, **kwargs
    ):
        if num_children is None:
            kwargs["num_children"] = num_children = len(children)
        if pos_offsets is None:
            kwargs["pos_offsets"] = np.stack(
                [
                    np.linspace(
                        -self.pos_offset_scale, self.pos_offset_scale, num=num_children
                    )
                ]
                * 2
            )
        return super()._unzip_setup_params(graph, parent, children, **kwargs)

    def _unzip_finalize(self, graph, **kwargs):
        super()._unzip_finalize(graph, **kwargs)
        self.update_positions(graph)

    def _press_finalize(self, graph, **kwargs):
        super()._unzip_finalize(graph, **kwargs)
        self.update_positions(graph)
