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

VertexID = NewType("VertexID", int)
ChildID = NewType("ChildID", VertexID)
ParentID = NewType("ParentID", VertexID)

SampleDepthScalar = NewType("SampleDepthScalar", float)
LengthScalar = NewType("LengthScalar", int)
SequenceScalar = NewType("SequenceScalar", str)

PropValueT = TypeVar("PropValueT")
UnzipParamT = TypeVar("UnzipParamT")
PressParamT = TypeVar("PressParamT")


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
        cls, parent_val: PropValueT, n: int, params: UnzipParamT
    ) -> Tuple[PropValueT, Sequence[PropValueT]]:
        # TODO: Docstring Output should be an sequence of PropValueT type, specifically.
        # Unzip at its most basic takes a parent value and copies it into the sequence of child values.
        # However, we also want to be able to handle the case of depth unzipping, where we need to also update
        # the parent depth (subtracting what got distributed to the children) and the distribution
        # to the children is parameterized.
        # For that reason we also accept a kwargs parameter, which parameterizes the unzipping into each child vertex.
        # The classmethod then returns both an updated value for the parent and each of the children.
        raise NotImplemented

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
        raise NotImplemented

    def __init__(self, vprop: VertexPropertyMap):
        self.vprop = vprop

    def unzip(self, parent: ParentID, children: Sequence[ChildID], params: UnzipParamT):
        """Unzip parent property value out to children."""
        curr_parent_val = self.vprop[parent]
        new_parent_val, new_child_vals = self.unzip_vals(
            curr_parent_val, len(children), params
        )
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
    def unzip_vals(cls, parent_val, n, params):
        out = np.empty(n, dtype=int)
        out[:] = parent_val
        return parent_val, [parent_val] * n

    @classmethod
    def press_vals(cls, parent_vals, params):
        return list(parent_vals), np.sum(parent_vals)


class SequenceProperty(ZipProperty[SequenceScalar, None, None]):
    def __init__(self, sequence):
        super().__init__(sequence)

    @classmethod
    def unzip_vals(cls, parent_val, n, params):
        return parent_val, [parent_val] * n

    @classmethod
    def press_vals(cls, parent_vals, params):
        out = SequenceScalar(",".join(parent_vals))
        return parent_vals, out


class DepthProperty(
    ZipProperty[SampleDepthScalar, Sequence[SampleDepthScalar], Sequence[LengthScalar]]
):
    def __init__(self, depth):
        super().__init__(depth)

    @classmethod
    def unzip_vals(cls, parent_val, n, params):
        return parent_val - np.sum(params), list(params)

    @classmethod
    def press_vals(cls, parent_vals, params):
        parent_vals = np.asarray(parent_vals)
        lengths = np.asarray(params)
        mean_depth = np.sum(parent_vals * lengths) / np.sum(params)
        return list(parent_vals - mean_depth), mean_depth


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
