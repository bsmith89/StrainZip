from typing import Generic, NewType, Sequence, Tuple, TypeVar

import numpy as np
from graph_tool import VertexPropertyMap

VertexID = NewType("VertexID", int)
ChildID = NewType("ChildID", VertexID)
ParentID = NewType("ParentID", VertexID)

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


class ZipProperty(Generic[PropValueT, UnzipParamT, PressParamT]):
    """Base class for property handling with unzip and press operations on a graph's vertices.

    This class should be subclassed to implement specific behavior for
    unzipping (expanding a parent property value to its children) and
    pressing (combining multiple property values into one).

    Attributes:
        vprop (VertexPropertyMap): The vertex property map from graph_tool to be manipulated.
    """

    @classmethod
    def unzip_vals(
        cls, parent_val: PropValueT, params: UnzipParamT
    ) -> Tuple[PropValueT, Sequence[PropValueT]]:
        """Class method to define how to unzip a parent value into child values.

        Subclasses must implement this method to specify custom unzipping logic.

        Args:
            parent_val (PropValueT): The value associated with the parent vertex.
            params (UnzipParamT): Parameters that affect how the unzipping is performed.

        Returns:
            A tuple containing the updated parent value and a sequence of values for the children.
        """
        raise NotImplementedError

    @classmethod
    def press_vals(
        cls, parent_vals: Sequence[PropValueT], params: PressParamT
    ) -> tuple[Sequence[PropValueT], PropValueT]:
        """Class method to define how to press child values into a single parent value.

        Subclasses must implement this method to specify custom pressing logic.

        Args:
            parent_vals (Sequence[PropValueT]): The values associated with the parent vertices.
            params (PressParamT): Parameters that affect how the pressing is performed.

        Returns:
            A tuple containing a sequence of updated parent values and the aggregated child value.
        """
        raise NotImplementedError

    def __init__(self, vprop: VertexPropertyMap):
        """Initializes the ZipProperty with a specific VertexPropertyMap."""
        self.vprop = vprop

    def unzip(self, parent: ParentID, children: Sequence[ChildID], params: UnzipParamT):
        """Performs the unzip operation from a parent vertex to its children.

        Args:
            parent (ParentID): The ID of the parent vertex.
            children (Sequence[ChildID]): A sequence of IDs for the child vertices.
            params (UnzipParamT): Parameters to guide the unzip operation.
        """
        curr_parent_val = self.vprop[parent]
        new_parent_val, new_child_vals = self.unzip_vals(curr_parent_val, params)
        self.vprop[parent] = new_parent_val
        for i, c in enumerate(children):
            self.vprop[c] = new_child_vals[i]

    def press(self, parents: Sequence[ParentID], child: ChildID, params: PressParamT):
        """Performs the press operation from parent vertices into a single child.

        Args:
            parents (Sequence[ParentID]): A sequence of IDs for the parent vertices.
            child (ChildID): The ID of the child vertex.
            params (PressParamT): Parameters to guide the press operation.
        """
        curr_parent_vals = [self.vprop[p] for p in parents]
        new_parent_vals, new_child_val = self.press_vals(curr_parent_vals, params)
        for i, p in enumerate(parents):
            self.vprop[p] = new_parent_vals[i]
        self.vprop[child] = new_child_val

    def batch_unzip(self, *args: Tuple[ParentID, Sequence[ChildID], UnzipParamT]):
        """Performs the unzip operation in batch for efficiency.

        This method can be overridden by subclasses for more efficient batch processing.

        Args:
            *args: Variable length argument list of tuples, each containing a parent ID,
                   a sequence of child IDs, and unzip parameters.
        """
        for parent, children, param in args:
            self.unzip(parent, children, param)

    def batch_press(self, *args: Tuple[Sequence[ParentID], ChildID, PressParamT]):
        """Performs the press operation in batch for efficiency.

        This method can be overridden by subclasses for more efficient batch processing.

        Args:
            *args: Variable length argument list of tuples, each containing a sequence of parent IDs,
                   a child ID, and press parameters.
        """
        for parents, child, param in args:
            self.press(parents, child, param)


class LengthProperty(ZipProperty[LengthScalar, None, None]):
    """Property class for handling lengths within the graph's vertices.

    This class specializes ZipProperty for length properties, facilitating
    the sharing and aggregation (sum) of length values among vertices.
    """

    @classmethod
    def unzip_vals(cls, parent_val, params=None):
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
    def press_vals(cls, parent_vals, params=None):
        """Implements the press operation for length properties.

        The children's values are aggregated (summed) into a single value.

        Args:
            parent_vals (Sequence[LengthScalar]): The length values of the parent vertices.

        Returns:
            A tuple containing a list of the original parent values and the sum of those values.
        """
        return list(parent_vals), np.sum(parent_vals)


class SequenceProperty(ZipProperty[SequenceScalar, None, None]):
    """Property class for handling sequences within the graph's vertices.

    This class specializes ZipProperty for sequence properties, facilitating
    the sharing and concatenation of sequence values among vertices.
    """

    @classmethod
    def unzip_vals(cls, parent_val, params=None):
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
    def press_vals(cls, parent_vals, params=None):
        """Implements the press operation for sequence properties.

        The children's values are concatenated into a single sequence value.

        Args:
            parent_vals (Sequence[SequenceScalar]): The sequence values of the parent vertices.

        Returns:
            A tuple containing a list of the original parent values and the concatenation of those values.
        """
        out = SequenceScalar(",".join(parent_vals))
        return parent_vals, out


class DepthProperty(
    ZipProperty[
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
    def unzip_vals(cls, parent_val, params):
        """Implements the unzip operation for depth properties.

        The parent value is reduced by the sum of the specified child depths.

        Args:
            parent_val (SampleDepthScalar): The depth value of the parent vertex.
            params (Sequence[SampleDepthScalar]): The depth values to distribute to the children.

        Returns:
            A tuple containing the residual parent depth after distribution to the children,
            and the sequence of child depths as specified in the parameters.
        """
        child_depths = np.asarray(params)
        parent_depth = np.asarray(parent_val)
        parent_depth = parent_depth.reshape((-1, 1))
        return parent_val - child_depths.sum(0), list(child_depths)

    @classmethod
    def press_vals(cls, parent_vals, params):
        """Implements the press operation for depth properties.

        The child's depth is calculated as the weighted mean of the parent depths,
        using the specified lengths as weights. Parents are adjusted to their residuals.

        Args:
            parent_vals (Sequence[SampleDepthScalar]): The depth values of the parent vertices.
            params (Sequence[LengthScalar]): The lengths to use as weights for the depth calculation.

        Returns:
            A tuple containing the list of adjusted parent depths and the weighted mean depth for the child.
        """
        lengths = np.asarray(params).reshape((-1, 1))
        num_parents = lengths.shape[0]
        parent_vals = np.asarray(parent_vals).reshape((num_parents, -1))
        mean_depth = np.sum(parent_vals * lengths, axis=0) / np.sum(lengths)
        return list(parent_vals - mean_depth), mean_depth


class PositionProperty(
    ZipProperty[PositionVector, Sequence[PositionVector], Sequence[LengthScalar]]
):
    """Property class for handling position values within the graph's vertices.

    This class specializes ZipProperty for position properties, facilitating
    the distribution and aggregation of position values among vertices based on specified offsets or weights.
    """

    @classmethod
    def unzip_vals(
        cls, parent_val, params
    ) -> Tuple[PositionVector, Sequence[PositionVector]]:
        """Implements the unzip operation for position properties.

        Children inherit the parent position, offset by an arbitrary amount specified in the parameters.

        Args:
            parent_val (PositionVector): The position value of the parent vertex.
            params (Sequence[PositionVector]): The offsets to apply to the parent value for each child.

        Returns:
            A tuple containing the unchanged parent position and a list of adjusted positions for the children.
        """
        child_offsets = np.asarray(params).reshape((2, -1))
        parent_val = np.asarray(parent_val).reshape((2, 1))
        # assert False
        # TODO: Figure out why reportIncompatibleMethodOverride
        child_vals = parent_val + child_offsets
        return parent_val, list(child_vals.T)  # type: ignore[reportIncompatibleMethodOverride]
        # TODO: Double check that list(child_depths) returns vectors of length 2 (x+y), not length nchildren.
        # TODO: Test by splitting into more than two children so nchildren != 2.

    @classmethod
    def press_vals(cls, parent_vals, params):
        """Implements the press operation for position properties.

        The child's position is the weighted mean of the parent position,
        using the specified params (nominally sequence lengths) as weights.

        Args:
            parent_vals (Sequence[PositionVector]): The position values of the parent vertices.
            params (Sequence[LengthScalar]): The lengths to use as weights for the position calculation.

        Returns:
            A tuple containing a list of the original parent positions and the weighted mean position for the child.
        """
        lengths = np.asarray(params).reshape((1, -1))
        parent_positions = np.asarray(parent_vals).reshape((-1, 2)).T
        mean_position = np.sum(parent_positions * lengths, axis=1) / np.sum(lengths)
        return list(parent_positions.T), mean_position


class FilterProperty(ZipProperty[bool, None, None]):
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
    def unzip_vals(cls, parent_val, params=None):
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
    def press_vals(cls, parent_vals, params=None):
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
