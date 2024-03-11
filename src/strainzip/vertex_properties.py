import math
from typing import Generic, NewType, Self, Sequence, Tuple, TypeVar

import graph_tool as gt
import numpy as np
from graph_tool import VertexPropertyMap

VertexID = NewType("VertexID", int)
ChildID = NewType("ChildID", VertexID)
ParentID = NewType("ParentID", VertexID)

SampleDepthScalar = NewType("SampleDepthScalar", float)
CoordinateScalar = NewType("CoordinateScalar", float)
SampleDepthVector = NewType("SampleDepthVector", Sequence[float])
LengthScalar = NewType("LengthScalar", int)
SequenceScalar = NewType("SequenceScalar", str)

PropValueT = TypeVar("PropValueT")
UnzipParamT = TypeVar("UnzipParamT")
PressParamT = TypeVar("PressParamT")


class InfiniteConstantSequence(Sequence[PropValueT]):
    def __init__(self, constant: PropValueT):
        self._constant: PropValueT = constant

    # Not clear why this method doesn't type check, but may be related to the
    # mutability of self.constant.
    def __getitem__(self, i) -> PropValueT:  # type: ignore[reportIncompatibleMethodOverride]
        return self._constant

    def __len__(self):
        raise NotImplementedError


class ZipProperty(Generic[PropValueT, UnzipParamT, PressParamT]):
    """Wraps a VertexPropertyMap with a notion of unzip and press.

    Handles the unzipping/pressing of vertex properties.

    Unzip takes one value and turns it into multiple.
    Press takes multiple values and turns them into one.

    Subclasses MUST implement custom classmethods for unzip_vals and press_vals.
    Subclasses MAY implement more efficient
    .unzip() / .press() / .batch_unzip() / .batch_press() methods.

    """

    @classmethod
    def unzip_vals(
        cls, parent_val: PropValueT, params: UnzipParamT
    ) -> Tuple[PropValueT, Sequence[PropValueT]]:
        # TODO: Docstring Output should be an sequence of PropValueT type, specifically.
        # Unzip at its most basic takes a parent value and copies it into the sequence of child values.
        # However, we also want to be able to handle the case of depth unzipping, where we need to also update
        # the parent depth (subtracting what got distributed to the children) and the distribution
        # to the children is parameterized.
        # For that reason we also accept a kwargs parameter, which parameterizes the unzipping into each child vertex.
        # The classmethod then returns both an updated value for the parent and each of the children.
        raise NotImplementedError

    @classmethod
    def press_vals(
        cls, parent_vals: Sequence[PropValueT], params: PressParamT
    ) -> tuple[Sequence[PropValueT], PropValueT]:
        # TODO: Docstring: Input should be an Sequence of PropValueT type, specifically.
        # Press at its most basic takes all the parent values and aggregates
        # (e.g. sum, mean, concatenation, etc.) them into a child value.
        # However, we also want to handle cases like depth pressing,
        # (where the parents all get a depth of 0 assigned after pressing)
        # Also, with depth pressing, it's not a mean, but a weighted mean,
        # weighted by the length, requiring an additional (vector) parameter.
        raise NotImplementedError

    def __init__(self, vprop: VertexPropertyMap):
        self.vprop = vprop

    def unzip(self, parent: ParentID, children: Sequence[ChildID], params: UnzipParamT):
        """Unzip parent property value out to children."""
        curr_parent_val = self.vprop[parent]
        new_parent_val, new_child_vals = self.unzip_vals(curr_parent_val, params)
        self.vprop[parent] = new_parent_val
        for i, c in enumerate(children):
            self.vprop[c] = new_child_vals[i]

    def press(self, parents: Sequence[ParentID], child: ChildID, params: PressParamT):
        """Press parent property values into child."""
        curr_parent_vals = [self.vprop[p] for p in parents]
        new_parent_vals, new_child_val = self.press_vals(curr_parent_vals, params)
        for i, p in enumerate(parents):
            self.vprop[p] = new_parent_vals[i]
        self.vprop[child] = new_child_val

    def batch_unzip(self, *args: Tuple[ParentID, Sequence[ChildID], UnzipParamT]):
        """Unzip multiple.

        Subclasses can override this method for efficiency.

        """
        for parent, children, param in args:
            self.unzip(parent, children, param)

    def batch_press(self, *args: Tuple[Sequence[ParentID], ChildID, PressParamT]):
        """Press multiple.

        Subclasses can override this method for efficiency.

        """
        for parents, child, param in args:
            self.press(parents, child, param)


class LengthProperty(ZipProperty[LengthScalar, None, None]):
    def __init__(self, length):
        super().__init__(length)

    @classmethod
    def unzip_vals(cls, parent_val, params):
        "Pass parent value through and share with children."
        return parent_val, InfiniteConstantSequence(parent_val)

    @classmethod
    def press_vals(cls, parent_vals, params=None):
        "Pass parent values through and sum into children."
        return list(parent_vals), np.sum(parent_vals)


class SequenceProperty(ZipProperty[SequenceScalar, None, None]):
    def __init__(self, sequence):
        super().__init__(sequence)

    @classmethod
    def unzip_vals(cls, parent_val, params=None):
        "Pass parent value through and share with children."
        return parent_val, InfiniteConstantSequence(parent_val)

    @classmethod
    def press_vals(cls, parent_vals, params=None):
        "Pass parent values through and concatenate into children."
        out = SequenceScalar(",".join(parent_vals))
        return parent_vals, out


class DepthProperty(
    ZipProperty[SampleDepthScalar, Sequence[SampleDepthScalar], Sequence[LengthScalar]]
):
    def __init__(self, depth):
        super().__init__(depth)

    @classmethod
    def unzip_vals(cls, parent_val, params):
        "Parent value becomes residual after summing arbitrary child values."
        child_depths = params
        return parent_val - np.sum(child_depths), child_depths

    @classmethod
    def press_vals(cls, parent_vals, params):
        "Child value is the weighted mean of arbitrary values; parents are the residuals."
        parent_vals = np.asarray(parent_vals)
        lengths = np.asarray(params)
        mean_depth = np.sum(parent_vals * lengths) / np.sum(params)
        return list(parent_vals - mean_depth), mean_depth


class CoordinateProperty(
    ZipProperty[CoordinateScalar, Sequence[CoordinateScalar], Sequence[LengthScalar]]
):
    def __init__(self, depth):
        super().__init__(depth)

    @classmethod
    def unzip_vals(cls, parent_val, params):
        "Pass parent value through; children are parent value offset by an arbitrary amount."
        child_offsets = np.asarray(params)
        return parent_val, list(parent_val + child_offsets)

    @classmethod
    def press_vals(cls, parent_vals, params=None):
        "Pass parent positions through; children are weighted mean of parents."
        lengths = np.asarray(params)
        parent_positions = np.asarray(parent_vals)
        mean_position = (parent_positions * lengths).sum() / lengths.sum()
        return list(parent_positions), mean_position
