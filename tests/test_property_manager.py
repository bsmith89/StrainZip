from typing import cast

import graph_tool as gt
import numpy as np

import strainzip as sz


def _pid(i: int) -> sz.types.ParentID:
    return cast(sz.types.ParentID, i)


def _cid(i: int) -> sz.types.ChildID:
    return cast(sz.types.ChildID, i)


def test_length_property():
    graph = gt.Graph()
    graph.add_edge_list([(0, 1), (1, 2), (2, 3)])
    graph.vertex_properties["length"] = graph.new_vertex_property(
        "int", vals=[1, 2, 3, 4]
    )

    # Value for all vertices initially set to 1.
    # print(list(graph.vertex_properties["length"]))
    assert np.array_equal(list(graph.vertex_properties["length"]), [1, 2, 3, 4])

    # After adding a new node, this value is the dtype default.
    graph.add_edge_list([(3, 4)])
    # print(list(graph.vertex_properties["length"]))
    assert np.array_equal(list(graph.vertex_properties["length"]), [1, 2, 3, 4, 0])

    graph.vertex_properties["length"][4] = 1
    # print(list(graph.vertex_properties["length"]))
    assert np.array_equal(list(graph.vertex_properties["length"]), [1, 2, 3, 4, 1])

    # "Unzip" vertex 2 into two new vertices.
    graph.add_vertex(n=2)
    sz.property_manager.length_manager.unzip(
        graph.vertex_properties["length"],
        _pid(2),
        [_cid(5), _cid(6)],
        num_children=2,
    )
    # print(list(graph.vertex_properties["length"]))
    assert np.array_equal(
        list(graph.vertex_properties["length"]), [1, 2, 3, 4, 1, 3, 3]
    )

    # "Pres" vertices 3+4 into a new vertex.
    graph.add_vertex()
    sz.property_manager.length_manager.press(
        graph.vertex_properties["length"], [_pid(3), _pid(4)], _cid(7), params=None
    )
    # print(list(graph.vertex_properties["length"]))
    assert np.array_equal(
        list(graph.vertex_properties["length"]), [1, 2, 3, 4, 1, 3, 3, 5]
    )


def test_seq_property():
    graph = gt.Graph()
    graph.add_edge_list([(0, 1), (1, 2), (2, 3)])
    graph.vertex_properties["seq"] = graph.new_vertex_property(
        "string", vals=["a", "b", "c", "d"]
    )
    assert np.array_equal(list(graph.vertex_properties["seq"]), ["a", "b", "c", "d"])

    # "Unzip" vertex 2 into two new vertices.
    graph.add_vertex(n=2)
    sz.property_manager.sequence_manager.unzip(
        graph.vertex_properties["seq"],
        _pid(2),
        [_cid(4), _cid(5)],
        num_children=2,
    )
    # print(list(graph.vertex_properties["seq"]))
    assert np.array_equal(
        list(graph.vertex_properties["seq"]), ["a", "b", "c", "d", "c", "c"]
    )

    # "Pres" vertices 0+1 into a new vertex.
    graph.add_vertex()
    sz.property_manager.sequence_manager.press(
        graph.vertex_properties["seq"], [_pid(0), _pid(1)], _cid(6), params=None
    )
    # print(list(graph.vertex_properties["seq"]))
    assert np.array_equal(
        list(graph.vertex_properties["seq"]), ["a", "b", "c", "d", "c", "c", "a,b"]
    )


def test_scalar_depth_property():
    graph = gt.Graph()
    graph.add_edge_list([(0, 1), (1, 2), (2, 3)])
    graph.vertex_properties["depth"] = graph.new_vertex_property(
        "float", vals=[1, 2, 3, 4]
    )
    assert np.array_equal(list(graph.vertex_properties["depth"]), [1, 2, 3, 4])

    # "Unzip" vertex 2 into two new vertices.
    graph.add_vertex(n=2)
    sz.property_manager.depth_manager.unzip(
        graph.vertex_properties["depth"],
        _pid(2),
        [_cid(4), _cid(5)],
        num_children=2,
        path_depths=[1.2, 1.3],
    )
    # print(list(graph.vertex_properties["depth"]))
    assert np.array_equal(
        list(graph.vertex_properties["depth"]), [1, 2, 0.5, 4, 1.2, 1.3]
    )

    # "Pres" vertices 0+1 into a new vertex.
    graph.add_vertex()
    sz.property_manager.depth_manager.press(
        graph.vertex_properties["depth"], [_pid(0), _pid(1)], _cid(6), lengths=[1, 2]
    )
    # print(list(graph.vertex_properties["depth"]))
    assert np.array_equal(
        list(graph.vertex_properties["depth"]),
        [
            -0.6666666666666667,
            0.33333333333333326,
            0.5,
            4.0,
            1.2,
            1.3,
            1.6666666666666667,
        ],
    )


def test_vector_depth_property():
    graph = gt.Graph()
    graph.add_edge_list([(0, 1), (1, 2), (2, 3)])
    graph.vertex_properties["depth"] = graph.new_vertex_property("vector<float>")
    graph.vertex_properties["depth"].set_2d_array(
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    )
    assert np.array_equal(
        graph.vertex_properties["depth"].get_2d_array(pos=[0, 1]),
        [[1, 2, 3, 4], [5, 6, 7, 8]],
    )

    # "Unzip" vertex 2 into two new vertices.
    graph.add_vertex(n=3)
    sz.property_manager.depth_manager.unzip(
        graph.vertex_properties["depth"],
        _pid(2),
        [_cid(4), _cid(5), _cid(6)],
        num_children=3,
        path_depths=[[1.2, 2], [1.3, 2], [0, 0]],
    )
    # print(list(graph.vertex_properties["depth"]))
    assert np.array_equal(
        graph.vertex_properties["depth"].get_2d_array(pos=[0, 1]),
        [
            [1, 2, 0.5, 4, 1.2, 1.3, 0],
            [5, 6, 3, 8, 2, 2, 0],
        ],
    )

    # "Pres" vertices 0+1 into a new vertex.
    graph.add_vertex()
    sz.property_manager.depth_manager.press(
        graph.vertex_properties["depth"], [_pid(0), _pid(1)], _cid(7), lengths=[2, 4]
    )
    # print(list(graph.vertex_properties["depth"]))
    assert np.allclose(
        graph.vertex_properties["depth"].get_2d_array(pos=[0, 1]),
        [
            [-0.66666667, 0.33333333, 0.5, 4.0, 1.2, 1.3, 0.0, 1.66666667],
            [-0.66666667, 0.33333333, 3.0, 8.0, 2.0, 2.0, 0.0, 5.66666667],
        ],
    )
