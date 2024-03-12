from typing import Callable, Generic, NewType, Sequence, Tuple, TypeVar

import numpy as np

from .types import ChildID, ParentID

__all__ = [
    "position_manager",
    "depth_manager",
    "length_manager",
    "filter_manager",
    "sequence_manager",
]

SampleDepthScalar = NewType("SampleDepthScalar", float)
PositionVector = NewType("PositionVector", Sequence[float])
SampleDepthVector = NewType("SampleDepthVector", Sequence[float])
LengthScalar = NewType("LengthScalar", int)
SequenceScalar = NewType("SequenceScalar", str)

PropValueT = TypeVar("PropValueT")
UnzipParamT = TypeVar("UnzipParamT")
PressParamT = TypeVar("PressParamT")


class InfiniteConstantSequence(Sequence[PropValueT]):
    """Represents a sequence where every element is the same constant value.

    This class provides an abstraction allowing for access to constant
    values as though they were sequences, without having to know the
    length ahead of time nor allocate an object.

    Attributes:
        _constant (PropValueT): The constant value returned for any index.
    """

    def __init__(self, constant: PropValueT):
        """Initialize the sequence with a constant value."""
        self._constant: PropValueT = constant

    # Not clear why this method doesn't type check, but may be related to the
    # mutability of self.constant.
    def __getitem__(self, i) -> PropValueT:  # type: ignore[reportIncompatibleMethodOverride]
        """Returns the constant value, regardless of the index."""
        return self._constant

    def __len__(self):
        """Raises NotImplementedError as the sequence is conceptually infinite."""
        raise NotImplementedError


class PropertyManager(Generic[PropValueT, UnzipParamT, PressParamT]):
    """Base class for property handling with unzip and press operations on a graph's vertices.

    This class should be subclassed to implement specific behavior for
    unzipping (expanding a parent property value to its children) and
    pressing (combining multiple property values into one).

    Attributes:
        vprop (VertexPropertyMap): The vertex property map from graph_tool to be manipulated.
    """

    @classmethod
    def from_namespace(cls, property_namespace):
        return cls(unzip=property_namespace.unzip, press=property_namespace.press)

    def __init__(
        self,
        unzip,
        press,
    ):
        """Initializes the ZipPropertyManager with functions for unzipping and pressing."""
        self.unzip_vals = unzip
        self.press_vals = press

    def unzip(
        self,
        vprop,
        parent: ParentID,
        children: Sequence[ChildID],
        **params: UnzipParamT
    ):
        """Performs the unzip operation from a parent vertex to its children.

        Args:
            parent (ParentID): The ID of the parent vertex.
            children (Sequence[ChildID]): A sequence of IDs for the child vertices.
            params (UnzipParamT): Parameters to guide the unzip operation.
        """
        curr_parent_val = vprop[parent]
        new_parent_val, new_child_vals = self.unzip_vals(curr_parent_val, **params)
        vprop[parent] = new_parent_val
        for i, c in enumerate(children):
            vprop[c] = new_child_vals[i]

    def press(
        self, vprop, parents: Sequence[ParentID], child: ChildID, **params: PressParamT
    ):
        """Performs the press operation from parent vertices into a single child.

        Args:
            parents (Sequence[ParentID]): A sequence of IDs for the parent vertices.
            child (ChildID): The ID of the child vertex.
            params (PressParamT): Parameters to guide the press operation.
        """
        curr_parent_vals = [vprop[p] for p in parents]
        new_parent_vals, new_child_val = self.press_vals(curr_parent_vals, **params)
        for i, p in enumerate(parents):
            vprop[p] = new_parent_vals[i]
        vprop[child] = new_child_val


class PropertyNamespace(Generic[PropValueT, UnzipParamT, PressParamT]):
    @classmethod
    def unzip(
        cls, parent_val: PropValueT, **params: UnzipParamT
    ) -> Tuple[PropValueT, Sequence[PropValueT]]:
        raise NotImplementedError

    @classmethod
    def press(
        cls, parent_vals: Sequence[PropValueT], **params: PressParamT
    ) -> Tuple[Sequence[PropValueT], PropValueT]:
        raise NotImplementedError


class Length(PropertyNamespace[LengthScalar, None, None]):
    """Property class for handling lengths within the graph's vertices.

    This class specializes ZipProperty for length properties, facilitating
    the sharing and aggregation (sum) of length values among vertices.
    """

    @classmethod
    def unzip(cls, parent_val, **params):
        """Implements the unzip operation for length properties.

        The parent value is passed through unchanged to all children.

        Args:
            parent_val (LengthScalar): The length value of the parent vertex.

        Returns:
            A tuple containing the unchanged parent value and an InfiniteConstantSequence
            of the parent value for the children.
        """
        return parent_val, InfiniteConstantSequence(parent_val)

    @classmethod
    def press(cls, parent_vals, **params):
        """Implements the press operation for length properties.

        The children's values are aggregated (summed) into a single value.

        Args:
            parent_vals (Sequence[LengthScalar]): The length values of the parent vertices.

        Returns:
            A tuple containing a list of the original parent values and the sum of those values.
        """
        return list(parent_vals), np.sum(parent_vals)


class Sequence_(PropertyNamespace[SequenceScalar, None, None]):
    """Property class for handling sequences within the graph's vertices.

    This class specializes ZipProperty for sequence properties, facilitating
    the sharing and concatenation of sequence values among vertices.
    """

    @classmethod
    def unzip(cls, parent_val, **params):
        """Implements the unzip operation for sequence properties.

        The parent value is passed through unchanged to all children.

        Args:
            parent_val (SequenceScalar): The sequence value of the parent vertex.

        Returns:
            A tuple containing the unchanged parent value and an InfiniteConstantSequence
            of the parent value for the children.
        """
        return parent_val, InfiniteConstantSequence(parent_val)

    @classmethod
    def press(cls, parent_vals, **params):
        """Implements the press operation for sequence properties.

        The children's values are concatenated into a single sequence value.

        Args:
            parent_vals (Sequence[SequenceScalar]): The sequence values of the parent vertices.

        Returns:
            A tuple containing a list of the original parent values and the concatenation of those values.
        """
        out = SequenceScalar(",".join(parent_vals))
        return parent_vals, out


class Depth(
    PropertyNamespace[
        SampleDepthScalar | SampleDepthVector,
        Sequence[SampleDepthScalar | SampleDepthVector],
        Sequence[LengthScalar],
    ]
):
    """Property class for handling depth values within the graph's vertices.

    This class specializes ZipProperty for depth properties, allowing for the
    distribution and aggregation of depth values based on arbitrary child and parent configurations.
    """

    @classmethod
    def unzip(cls, parent_val, path_depths=None, **params):
        """Implements the unzip operation for depth properties.

        The parent value is reduced by the sum of the specified child depths.

        Args:
            parent_val (SampleDepthScalar): The depth value of the parent vertex.
            params (Sequence[SampleDepthScalar]): The depth values to distribute to the children.

        Returns:
            A tuple containing the residual parent depth after distribution to the children,
            and the sequence of child depths as specified in the parameters.
        """
        assert path_depths is not None
        child_depths = np.asarray(path_depths)
        parent_depth = np.asarray(parent_val)
        parent_depth = parent_depth.reshape((-1, 1))
        return parent_val - child_depths.sum(0), list(child_depths)

    @classmethod
    def press(cls, parent_vals, lengths=None, **params):
        """Implements the press operation for depth properties.

        The child's depth is calculated as the weighted mean of the parent depths,
        using the specified lengths as weights. Parents are adjusted to their residuals.

        Args:
            parent_vals (Sequence[SampleDepthScalar]): The depth values of the parent vertices.
            params (Sequence[LengthScalar]): The lengths to use as weights for the depth calculation.

        Returns:
            A tuple containing the list of adjusted parent depths and the weighted mean depth for the child.
        """
        assert lengths is not None
        lengths = np.asarray(lengths).reshape((-1, 1))
        num_parents = lengths.shape[0]
        parent_vals = np.asarray(parent_vals).reshape((num_parents, -1))
        mean_depth = np.sum(parent_vals * lengths, axis=0) / np.sum(lengths)
        return list(parent_vals - mean_depth), mean_depth


class Position(
    PropertyNamespace[PositionVector, Sequence[PositionVector], Sequence[LengthScalar]]
):
    """Property class for handling position values within the graph's vertices.

    This class specializes ZipProperty for position properties, facilitating
    the distribution and aggregation of position values among vertices based on specified offsets or weights.
    """

    @classmethod
    def unzip(
        cls, parent_val, pos_offsets=None, **params
    ) -> Tuple[PositionVector, Sequence[PositionVector]]:
        """Implements the unzip operation for position properties.

        Children inherit the parent position, offset by an arbitrary amount specified in the parameters.

        Args:
            parent_val (PositionVector): The position value of the parent vertex.
            params (Sequence[PositionVector]): The offsets to apply to the parent value for each child.

        Returns:
            A tuple containing the unchanged parent position and a list of adjusted positions for the children.
        """
        assert pos_offsets is not None
        child_offsets = np.asarray(pos_offsets).reshape((2, -1))
        parent_val = np.asarray(parent_val).reshape((2, 1))
        child_vals = parent_val + child_offsets
        # TODO: Figure out why reportIncompatibleMethodOverride
        return parent_val, list(child_vals.T)  # type: ignore[reportIncompatibleMethodOverride]

    @classmethod
    def press(cls, parent_vals, lengths=None, **params):
        """Implements the press operation for position properties.

        The child's position is the weighted mean of the parent position,
        using the specified params (nominally sequence lengths) as weights.

        Args:
            parent_vals (Sequence[PositionVector]): The position values of the parent vertices.
            params (Sequence[LengthScalar]): The lengths to use as weights for the position calculation.

        Returns:
            A tuple containing a list of the original parent positions and the weighted mean position for the child.
        """
        assert lengths is not None
        lengths = np.asarray(lengths).reshape((1, -1))
        parent_positions = np.asarray(parent_vals).reshape((-1, 2)).T
        mean_position = np.sum(parent_positions * lengths, axis=1) / np.sum(lengths)
        return list(parent_positions.T), mean_position


class Filter(PropertyNamespace[bool, None, None]):
    """
    A specialized property for handling boolean filter operations on vertices within a graph.

    This class extends ZipProperty for boolean values, specifically designed to simplify
    filter-like operations where a parent's value can be 'unzipped' into its children,
    and children's values can be 'pressed' back into a parent. The unique aspect of this
    property is its simplistic approach to these operations, tailored to boolean contexts.

    The `unzip_vals` method always generates a constant sequence of `True` values for children,
    regardless of the parent's original value, effectively 'filtering in' all children.
    The `press_vals` method, conversely, sets all parent values to `False` and yields `True` for
    the resultant child, indicating a default inclusion criterion for the press operation.

    This property can be particularly useful in scenarios where a boolean condition needs to
    be applied across a hierarchy or a structure of vertices, simplifying the propagation of
    filter criteria.
    """

    @classmethod
    def unzip(cls, parent_val, **params):
        """
        Unzips a boolean value from a parent to its children, setting all children to `True`.

        This method overrides the `unzip_vals` method in ZipProperty to provide specific
        behavior for boolean properties, ignoring the actual parent value and instead
        unconditionally enabling (setting to `True`) all child vertices.

        Args:
            parent_val (bool): The boolean value associated with the parent vertex. This
                               value is ignored in the current implementation.
            params (None): Parameters for the unzip operation. Currently not used.

        Returns:
            A tuple consisting of `False` for the updated parent value (indicating the
            parent is 'filtered out' post-unzip) and an `InfiniteConstantSequence` of `True`
            for all child vertices (indicating all children are 'filtered in').
        """
        return False, InfiniteConstantSequence(True)

    @classmethod
    def press(cls, parent_vals, **params):
        """
        Presses boolean values from parents to a child, setting the child to `True`.

        This method overrides the `press_vals` method in ZipProperty to provide specific
        behavior for boolean properties. It sets the resultant value for the child vertex
        to `True` unconditionally, while marking all parent vertices as `False`, indicating
        they are 'filtered out' after the press operation.

        Args:
            parent_vals (Sequence[bool]): A sequence of boolean values associated with the
                                          parent vertices. Besides their length, these values are ignored in the
                                          current implementation.
            params (None): Parameters for the press operation. Currently not used.

        Returns:
            A tuple consisting of a list of `False` for each parent (indicating they are
            'filtered out') and `True` for the resultant child value (indicating the child
            is 'filtered in').
        """
        return [False] * len(parent_vals), True


position_manager = PropertyManager.from_namespace(Position)
sequence_manager = PropertyManager.from_namespace(Sequence_)
length_manager = PropertyManager.from_namespace(Length)
depth_manager = PropertyManager.from_namespace(Depth)
filter_manager = PropertyManager.from_namespace(Filter)
