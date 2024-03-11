from typing import cast

import graph_tool as gt
import numpy as np

from strainzip.vertex_properties import (
    ChildID,
    DepthProperty,
    LengthProperty,
    LengthScalar,
    ParentID,
    SampleDepthScalar,
    SampleDepthVector,
    SequenceProperty,
)


def _pid(i: int) -> ParentID:
    return cast(ParentID, i)


def _cid(i: int) -> ChildID:
    return cast(ChildID, i)


def test_length_property():
    g = gt.Graph()

    g.add_edge_list([(0, 1)])

    p0 = g.new_vertex_property("int", val=1)
    assert np.array_equal(
        list(p0), [1, 1]
    )  # Value for all vertices initially set to 1.

    p0[1] = 2
    assert np.array_equal(list(p0), [1, 2])

    g.add_edge_list(
        [(0, 2)]
    )  # New vertex (added implicitly by new edge) has "default" value of 0 (not 1).
    assert np.array_equal(list(p0), [1, 2, 0])

    p0.a[:2] = 3
    assert np.array_equal(list(p0), [3, 3, 0])

    # Add vertices with index 3 and 4.
    g.add_vertex(n=2)
    p1 = LengthProperty(p0)
    assert np.array_equal(list(p1.vprop), [3, 3, 0, 0, 0])

    # "Unzip" the length value of node 0 into nodes 3 and 4
    p1.unzip(_pid(0), [_cid(3), _cid(4)], params=None)
    assert np.array_equal(list(p1.vprop), [3, 3, 0, 3, 3])

    # Add a new vertex
    g.add_vertex()
    # And "Press" the length value of nodes [0, 1] into node 5
    p1.press([_pid(0), _pid(1)], _cid(5), params=None)
    assert np.array_equal(list(p1.vprop), [3, 3, 0, 3, 3, 6])


def test_sequence_property():
    g = gt.Graph()

    g.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4)])

    p0 = g.new_vertex_property("string")
    assert np.array_equal(list(p0), ["", "", "", "", ""])

    p0[0] = "0"
    assert np.array_equal(list(p0), ["0", "", "", "", ""])

    for i in range(len(list(p0))):
        p0[i] = str(i)
    assert np.array_equal(list(p0), ["0", "1", "2", "3", "4"])

    p1 = SequenceProperty(p0)
    assert np.array_equal(list(p1.vprop), ["0", "1", "2", "3", "4"])

    # Add vertices with index 5 and 6.
    g.add_vertex(n=2)
    assert np.array_equal(list(p1.vprop), ["0", "1", "2", "3", "4", "", ""])

    # "Unzip" the sequence value of node 0 into nodes [5, 6]
    p1.unzip(_pid(0), [_cid(5), _cid(6)], params=None)
    assert np.array_equal(list(p1.vprop), ["0", "1", "2", "3", "4", "0", "0"])

    # Add a new vertex
    g.add_vertex()
    # And "Press" the sequence value of nodes [0, 1] into node 7
    p1.press([_pid(0), _pid(1)], _cid(7), params=None)
    assert np.array_equal(list(p1.vprop), ["0", "1", "2", "3", "4", "0", "0", "0,1"])


def test_depth_property():
    g = gt.Graph()

    g.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4)])

    p0 = g.new_vertex_property("float")
    assert np.allclose(
        list(p0), [0.0, 0.0, 0.0, 0.0, 0.0]
    )  # Value for all vertices initially set to 1.

    p0[0] = 0.1
    assert np.allclose(list(p0), [0.1, 0.0, 0.0, 0.0, 0.0])

    for i in range(len(list(p0))):
        p0[i] = np.sqrt(5 + i)
    assert np.allclose(
        list(p0),
        [
            2.23606797749979,
            2.449489742783178,
            2.6457513110645907,
            2.8284271247461903,
            3.0,
        ],
    )

    p1 = DepthProperty(p0)
    assert np.allclose(
        list(p1.vprop),
        [
            2.23606797749979,
            2.449489742783178,
            2.6457513110645907,
            2.8284271247461903,
            3.0,
        ],
    )

    # Add vertices with index 5 and 6.
    g.add_vertex(n=2)
    assert np.allclose(
        list(p1.vprop),
        [
            2.23606797749979,
            2.449489742783178,
            2.6457513110645907,
            2.8284271247461903,
            3.0,
            0.0,
            0.0,
        ],
    )

    # "Unzip" the depth value of node 0 into nodes [5, 6]
    p1.unzip(
        _pid(0),
        [_cid(5), _cid(6)],
        params=[SampleDepthScalar(1.0), SampleDepthScalar(1.2)],
    )
    assert np.allclose(
        list(p1.vprop),
        [
            0.03606797749978963,
            2.449489742783178,
            2.6457513110645907,
            2.8284271247461903,
            3.0,
            1.0,
            1.2,
        ],
    )

    # Add a new vertex
    g.add_vertex()
    # And "Press" the depth value of nodes [0, 1] into node 7
    p1.press([_pid(3), _pid(4)], _cid(7), params=[LengthScalar(2), LengthScalar(5)])
    assert np.allclose(
        list(p1.vprop),
        [
            0.03606797749978963,
            2.449489742783178,
            2.6457513110645907,
            -0.12255205375272116,
            0.04902082150108855,
            1.0,
            1.2,
            2.9509791784989114,
        ],
    )
