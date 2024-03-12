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


class BasePropertyManager:
    @classmethod
    def from_namespace(cls, property_namespace):
        return cls(unzip=property_namespace.unzip, press=property_namespace.press)

    def __init__(
        self,
        unzip,
        press,
    ):
        self.unzip_vals = unzip
        self.press_vals = press

    def unzip(
        self,
        vprop,
        parent: ParentID,
        children: Sequence[ChildID],
        **params,
    ):
        curr_parent_val = vprop[parent]
        new_parent_val, new_child_vals = self.unzip_vals(curr_parent_val, **params)
        vprop[parent] = new_parent_val
        for i, c in enumerate(children):
            vprop[c] = new_child_vals[i]

    def press(self, vprop, parents: Sequence[ParentID], child: ChildID, **params):
        curr_parent_vals = [vprop[p] for p in parents]
        new_parent_vals, new_child_val = self.press_vals(curr_parent_vals, **params)
        for i, p in enumerate(parents):
            vprop[p] = new_parent_vals[i]
        vprop[child] = new_child_val

    def batch_unzip(self, vprop, *args):
        for parent, children, params in args:
            self.unzip(vprop, parent, children, **params)

    def batch_press(self, vprop, *args):
        for parents, child, params in args:
            self.press(vprop, parents, child, **params)


class ArrayablePropertyManager(BasePropertyManager):
    def unzip(
        self,
        vprop,
        parent: ParentID,
        children: Sequence[ChildID],
        **params,
    ):
        curr_parent_val = vprop.a[parent]
        new_parent_val, new_child_vals = self.unzip_vals(curr_parent_val, **params)
        vprop.a[parent] = new_parent_val
        vprop.a[children] = new_child_vals

    def press(
        self,
        vprop,
        parents: Sequence[ParentID],
        child: ChildID,
        **params,
    ):
        curr_parent_vals = vprop.a[parents]
        new_parent_vals, new_child_val = self.press_vals(curr_parent_vals, **params)
        vprop.a[parents] = new_parent_vals
        vprop.a[child] = new_child_val


class PropertyNamespace:
    @classmethod
    def unzip(cls, parent_val, **params):
        raise NotImplementedError

    @classmethod
    def press(cls, parent_vals, **params):
        raise NotImplementedError


class Length(PropertyNamespace):
    @classmethod
    def unzip(cls, parent_val, num_children=None, **params):
        assert num_children is not None
        return parent_val, np.ones(num_children, dtype=int) * parent_val

    @classmethod
    def press(cls, parent_vals, **params):
        return list(parent_vals), np.sum(parent_vals)


class Sequence_(PropertyNamespace):
    @classmethod
    def unzip(cls, parent_val, num_children=None, **params):
        assert num_children is not None
        return parent_val, np.array([parent_val] * num_children, dtype=str)

    @classmethod
    def press(cls, parent_vals, **params):
        out = SequenceScalar(",".join(parent_vals))
        return parent_vals, out


class Depth(PropertyNamespace):
    @classmethod
    def unzip(cls, parent_val, path_depths=None, **params):
        assert path_depths is not None
        child_depths = np.asarray(path_depths)
        parent_depth = np.asarray(parent_val)
        parent_depth = parent_depth.reshape((-1, 1))
        return parent_val - child_depths.sum(0), list(child_depths)

    @classmethod
    def press(cls, parent_vals, lengths=None, **params):
        assert lengths is not None
        lengths = np.asarray(lengths).reshape((-1, 1))
        num_parents = lengths.shape[0]
        parent_vals = np.asarray(parent_vals).reshape((num_parents, -1))
        mean_depth = np.sum(parent_vals * lengths, axis=0) / np.sum(lengths)
        return list(parent_vals - mean_depth), mean_depth


class Position(PropertyNamespace):
    @classmethod
    def unzip(cls, parent_val, pos_offsets=None, **params):
        assert pos_offsets is not None
        child_offsets = np.asarray(pos_offsets).reshape((2, -1))
        parent_val = np.asarray(parent_val).reshape((2, 1))
        child_vals = parent_val + child_offsets
        # TODO: Figure out why reportIncompatibleMethodOverride
        return parent_val, list(child_vals.T)  # type: ignore[reportIncompatibleMethodOverride]

    @classmethod
    def press(cls, parent_vals, lengths=None, **params):
        assert lengths is not None
        lengths = np.asarray(lengths).reshape((1, -1))
        parent_positions = np.asarray(parent_vals).reshape((-1, 2)).T
        mean_position = np.sum(parent_positions * lengths, axis=1) / np.sum(lengths)
        return list(parent_positions.T), mean_position


class Filter(PropertyNamespace):
    @classmethod
    def unzip(cls, parent_val, num_children=None, **params):
        assert num_children is not None
        return False, np.ones(num_children, dtype=bool)

    @classmethod
    def press(cls, parent_vals, **params):
        return [False] * len(parent_vals), True


position_manager = BasePropertyManager.from_namespace(Position)
sequence_manager = BasePropertyManager.from_namespace(Sequence_)
length_manager = ArrayablePropertyManager.from_namespace(Length)
depth_manager = BasePropertyManager.from_namespace(Depth)
filter_manager = ArrayablePropertyManager.from_namespace(Filter)
