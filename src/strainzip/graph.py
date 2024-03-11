from typing import Any, Optional, Self, Sequence, Tuple, TypeAlias, cast

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
    """Base class for handling graph properties through the ZipProperty paradigm.

    This class manages a graph and its associated ZipProperty instances,
    orchestrating the unzip and press operations across all properties.

    Attributes:
        graph (gt.Graph): The underlying graph_tool graph instance.
        props (dict[str, ZipProperty]): A dictionary mapping property names to their respective ZipProperty instances.
    """

    def __init__(self, graph: gt.Graph, **extra_props: ZipProperty):
        """Initializes a BaseZipGraph with a graph and optional extra properties.

        Args:
            graph (gt.Graph): The graph to which the properties are associated.
            **extra_props: Arbitrary keyword arguments for additional ZipProperty instances.
        """
        self.graph = graph
        self.props = extra_props

    def unzip(
        self,
        parent: ParentID,
        paths: Sequence[Tuple[VertexID, VertexID]],
        **extra_params: AnyUnzipParam,
    ):
        """Performs the unzip operation on the graph, adding new vertices and adjusting properties accordingly.

        Args:
            parent (ParentID): The ID of the parent vertex from which to unzip.
            paths (Sequence[Tuple[VertexID, VertexID]]): A sequence of tuples representing the paths for unzipping.
            **extra_params: Arbitrary keyword parameters for unzip parameters specific to each property.
        """
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
        """Performs the press operation on the graph, combining vertices and adjusting properties accordingly.

        Args:
            parents (Sequence[ParentID]): A sequence of parent IDs whose properties are to be pressed into a new child vertex.
            **extra_params: Arbitrary keyword parameters for press parameters specific to each property.
        """
        # Implementation omitted for brevity
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
    """A specialized ZipGraph that handles length and sequence properties in addition to an automatic filter.

    Inherits from BaseZipGraph and adds specific handling for length and sequence properties.

    Attributes:
        length (LengthProperty): The length property associated with the graph's vertices.
        sequence (SequenceProperty): The sequence property associated with the graph's vertices.
        filter (FilterProperty): An automatic boolean filter property for vertices.
    """

    def __init__(
        self: Self,
        graph: gt.Graph,
        length: LengthProperty,
        sequence: SequenceProperty,
        **extra_props: ZipProperty,
    ):
        """Initializes a SequenceZipGraph with length and sequence properties.

        Args:
            graph (gt.Graph): The underlying graph_tool graph instance.
            length (LengthProperty): The length property for the graph's vertices.
            sequence (SequenceProperty): The sequence property for the graph's vertices.
            **extra_props: Arbitrary keyword arguments for additional ZipProperty instances.
        """
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
    """Extends SequenceZipGraph to include depth handling, creating a comprehensive graph property management system.

    Attributes:
        depth (DepthProperty): The depth property associated with the graph's vertices.
    """

    def __init__(
        self: Self,
        graph: gt.Graph,
        length: LengthProperty,
        sequence: SequenceProperty,
        depth: DepthProperty,
        **extra_props: ZipProperty,
    ):
        """Initializes a DepthZipGraph with length, sequence, and depth properties.

        Args:
            graph (gt.Graph): The underlying graph_tool graph instance.
            length (LengthProperty): The length property for the graph's vertices.
            sequence (SequenceProperty): The sequence property for the graph's vertices.
            depth (DepthProperty): The depth property for the graph's vertices.
            **extra_props: Additional ZipProperty instances.
        """
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
    """Incorporates visualization capabilities into DepthZipGraph by managing vertex positions for graphical display.

    Attributes:
        xyposition (PositionProperty): The property for storing the x and y coordinates of each vertex.
        pos_offset_scale (float): Scale factor for adjusting
        the offsets during the unzip operation to position children vertices.
        sfdp_layout_kwargs (Optional[dict[str, Any]]): Parameters for the sfdp layout algorithm used in updating vertex positions.

    This class extends DepthZipGraph by adding a position property and methods to automatically update vertex positions after graph operations, facilitating visualization of the graph's structure.
    """

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
        """Initializes a VizZipGraph with properties for depth, length, sequence, and vertex positions.

        Args:
            graph (gt.Graph): The graph on which operations are performed.
            depth (DepthProperty): A ZipProperty for managing depth information.
            length (LengthProperty): A ZipProperty for managing length information.
            sequence (SequenceProperty): A ZipProperty for managing sequence information.
            xyposition (PositionProperty): A ZipProperty for managing the x and y coordinates of vertices.
            pos_offset_scale (float, optional): Scale for adjusting position offsets. Defaults to 0.1.
            sfdp_layout_kwargs (Optional[dict[str, Any]], optional): Additional arguments for the sfdp layout algorithm. Defaults to None.
            **extra_props: Additional ZipProperty instances for extending functionality.
        """
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
        self.update_positions()

    def update_positions(self, **kwargs):
        """Updates vertex positions using the sfdp layout algorithm with the current position property as an initial layout.

        Args:
            **kwargs: Optional arguments to override sfdp layout parameters for this update.
        """
        sfdp_layout_kwargs = self.sfdp_layout_kwargs | kwargs
        xyposition = gtdraw.sfdp_layout(
            self.graph,
            pos=self.props["xyposition"].vprop,
            **sfdp_layout_kwargs,
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
        """Extends the unzip operation with automatic position updates for new vertices.

        Args:
            parent: The ID of the parent vertex.
            paths: A sequence of tuples specifying the paths for unzipping.
            **extra_params: Additional parameters for customizing the unzip operation.
        """
        coord_offsets = np.linspace(
            -self.pos_offset_scale, self.pos_offset_scale, num=len(paths)
        )
        coord_offsets = np.stack([coord_offsets] * 2)

        params = dict(xyposition=coord_offsets) | extra_params
        super().unzip(parent, paths, **params)
        self.update_positions()

    def press(self, parents, **extra_params):
        """Extends the press operation with automatic position updates for the new vertex.

        Args:
            parents: A sequence of parent IDs whose properties are to be pressed into a new child vertex.
            **extra_params: Additional parameters for customizing the press operation.
        """
        params = (
            dict(
                xyposition=self.props["length"].vprop.a[parents],
            )
            | extra_params
        )
        super().press(parents, **params)
        self.update_positions()
