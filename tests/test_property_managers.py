from typing import cast

import graph_tool as gt
import numpy as np

import strainzip as sz


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
    length_unzipper = sz.graph_manager.LengthUnzipper()
    length_unzipper.unzip(
        graph,
        2,
        [5, 6],
    )
    # print(list(graph.vertex_properties["length"]))
    assert np.array_equal(
        list(graph.vertex_properties["length"]), [1, 2, 3, 4, 1, 3, 3]
    )

    # "Pres" vertices 3+4 into a new vertex.
    graph.add_vertex()
    length_presser = sz.graph_manager.LengthPresser()
    length_presser.press(
        graph,
        [3, 4],
        7,
    )
    # print(list(graph.vertex_properties["length"]))
    assert np.array_equal(
        list(graph.vertex_properties["length"]), [1, 2, 3, 4, 1, 3, 3, 5]
    )


def test_sequence_property():
    graph = gt.Graph()
    graph.add_edge_list([(0, 1), (1, 2), (2, 3)])
    graph.vertex_properties["sequence"] = graph.new_vertex_property(
        "string", vals=["a", "b", "c", "d"]
    )
    assert np.array_equal(
        list(graph.vertex_properties["sequence"]), ["a", "b", "c", "d"]
    )

    # "Unzip" vertex 2 into two new vertices.
    graph.add_vertex(n=2)
    sequence_unzipper = sz.graph_manager.SequenceUnzipper()
    sequence_unzipper.unzip(
        graph,
        2,
        [4, 5],
    )
    # print(list(graph.vertex_properties["sequence"]))
    assert np.array_equal(
        list(graph.vertex_properties["sequence"]), ["a", "b", "c", "d", "c", "c"]
    )

    # "Pres" vertices 0+1 into a new vertex.
    graph.add_vertex()
    sequence_presser = sz.graph_manager.SequencePresser()
    sequence_presser.press(
        graph,
        [0, 1],
        6,
    )
    # print(list(graph.vertex_properties["sequence"]))
    assert np.array_equal(
        list(graph.vertex_properties["sequence"]), ["a", "b", "c", "d", "c", "c", "a,b"]
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
    depth_unzipper = sz.graph_manager.ScalarDepthUnzipper()
    depth_unzipper.unzip(
        graph,
        2,
        [4, 5],
        ([1.2, 1.3],),
    )
    # print(list(graph.vertex_properties["depth"]))
    assert np.array_equal(
        list(graph.vertex_properties["depth"]), [1, 2, 0.5, 4, 1.2, 1.3]
    )

    # "Pres" vertices 0+1 into a new vertex.
    graph.add_vertex()
    # ScalarDepthPresser requires graph to have a length property.
    graph.vertex_properties["length"] = graph.new_vertex_property(
        "int", vals=[1, 2, 1, 2, 1, 2, 0]
    )
    depth_presser = sz.graph_manager.ScalarDepthPresser()
    depth_presser.press(
        graph,
        [0, 1],
        6,
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
    graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4)])
    graph.vertex_properties["depth"] = graph.new_vertex_property("vector<float>")
    graph.vertex_properties["depth"].set_2d_array(
        np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    )
    assert np.array_equal(
        graph.vertex_properties["depth"].get_2d_array(pos=[0, 1]),
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    )

    # "Unzip" vertex 2 into two new vertices.
    graph.add_vertex(n=3)
    depth_unzipper = sz.graph_manager.VectorDepthUnzipper()
    depth_unzipper.unzip(
        graph,
        3,
        [5, 6, 7],
        ([[1.2, 2], [1.3, 2], [0, 0]],),
    )
    # print(list(graph.vertex_properties["depth"]))
    assert np.array_equal(
        graph.vertex_properties["depth"].get_2d_array(pos=[0, 1]),
        [
            [0, 1, 2, 0.5, 4, 1.2, 1.3, 0],
            [5, 6, 7, 4, 9, 2, 2, 0],
        ],
    )

    # "Pres" vertices 0+1 into a new vertex.
    graph.add_vertex()
    # VectorDepthPresser requires graph to have a length property.
    graph.vp["length"] = graph.new_vertex_property("int", vals=[2, 4, 1, 1, 1, 1, 1])
    depth_presser = sz.graph_manager.VectorDepthPresser()
    depth_presser.press(graph, [0, 1, 2], 8)
    # print(list(graph.vertex_properties["depth"]))
    assert np.allclose(
        graph.vertex_properties["depth"].get_2d_array(pos=[0, 1]),
        [
            [-0.85714286, 0.14285714, 1.14285714, 0.5, 4.0, 1.2, 1.3, 0.0, 0.85714286],
            [-0.85714286, 0.14285714, 1.14285714, 4.0, 9.0, 2.0, 2.0, 0.0, 5.85714286],
        ],
    )
