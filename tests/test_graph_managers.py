from contextlib import contextmanager
from typing import cast

import graph_tool as gt
import graph_tool.draw
import numpy as np

import strainzip as sz


@contextmanager
def unfiltered(graph):
    filt = graph.get_vertex_filter()
    graph.set_vertex_filter(None)
    yield
    graph.set_vertex_filter(*filt)


def test_batch_unzip_topology_problematic():
    # When this batch unzip works correctly...
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 5), (1, 5), (5, 2), (5, 6), (6, 3), (6, 4)])
    _graph.vp["filter"] = _graph.new_vertex_property("bool", val=True)
    _graph.set_vertex_filter(_graph.vp["filter"])
    gm = sz.graph_manager.GraphManager()
    gm.validate(_graph)
    gm.batch_unzip(
        _graph,
        (6, [(5, 3), (5, 4)], {}),
        (5, [(0, 2), (1, 6)], {}),
    )
    assert np.array_equal(
        sz.stats.degree_stats(_graph).sort_index().reset_index().values,
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [1.0, 1.0, 3.0],
            [1.0, 2.0, 1.0],
        ],
    )

    # Swapping the order of the batch unzip breaks things:
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 5), (1, 5), (5, 2), (5, 6), (6, 3), (6, 4)])
    # sz.draw.draw_graph(_graph, ink_scale=0.35, vertex_text=_graph.vertex_index)
    _graph.vp["filter"] = _graph.new_vertex_property("bool", val=True)
    _graph.set_vertex_filter(_graph.vp["filter"])
    gm = sz.graph_manager.GraphManager()
    gm.validate(_graph)
    gm.batch_unzip(
        _graph,
        (5, [(0, 2), (1, 6)], {}),
        (6, [(5, 3), (5, 4)], {}),
    )
    # sz.draw.draw_graph(_graph, ink_scale=0.35, vertex_text=_graph.vertex_index)
    assert np.array_equal(
        sz.stats.degree_stats(_graph).sort_index().reset_index().values,
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [1.0, 1.0, 3.0],
            [1.0, 2.0, 1.0],
        ],
    )


def test_graph_positioning():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    num_vertices = _graph.num_vertices(ignore_filter=True)
    # TODO: Assign depths to _depth
    _length = _graph.new_vertex_property("int", vals=[1, 2, 1, 2, 1, 2])
    _sequence = _graph.new_vertex_property("string", vals=range(len(_graph)))

    depth = np.array(
        [
            [0, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ]
    )
    _depth = _graph.new_vertex_property("vector<float>")
    _depth.set_2d_array(depth, pos=[0, 1])

    offset_scale = 0.1
    xyposition = np.empty((2, num_vertices))
    xyposition[0, :] = np.arange(num_vertices)
    xyposition[1, :] = 0
    _xyposition = _graph.new_vertex_property("vector<float>")
    _xyposition.set_2d_array(xyposition, pos=[0, 1])

    _filter = _graph.new_vertex_property("bool", val=1)

    _graph.vp["depth"] = _depth
    _graph.vp["length"] = _length
    _graph.vp["sequence"] = _sequence
    _graph.vp["xyposition"] = _xyposition
    _graph.vp["filter"] = _filter

    _graph.set_vertex_filter(_graph.vp["filter"])

    gm = sz.graph_manager.GraphManager(
        unzippers=[
            sz.graph_manager.LengthUnzipper(),
            sz.graph_manager.SequenceUnzipper(),
            sz.graph_manager.VectorDepthUnzipper(),
            sz.graph_manager.PositionUnzipper(offset=(0.1, 0.1)),
        ],
        pressers=[
            sz.graph_manager.LengthPresser(),
            sz.graph_manager.SequencePresser(sep=","),
            sz.graph_manager.VectorDepthPresser(),
            sz.graph_manager.PositionPresser(),
        ],
    )
    gm.validate(_graph)

    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['xyposition'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    with unfiltered(_graph):
        assert np.allclose(
            _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
            np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        )

    gm.unzip(
        _graph,
        2,
        [(1, 3), (1, 3), (1, 3)],
        path_depths=[[0.33, 0], [0.33, 0], [0.33, 0.2]],
    )
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['xyposition'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    with unfiltered(_graph):
        assert np.allclose(
            _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
            np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.85, 2.0, 2.15],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0, 0.15],
                ]
            ),
        )

    gm.unzip(_graph, 8, [(1, 3), (1, 3)], path_depths=[[0.3, 0], [0.1, 0]])
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['xyposition'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    with unfiltered(_graph):
        assert np.allclose(
            _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
            np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.85, 2.0, 2.15, 2.05, 2.25],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0, 0.15, 0.05, 0.25],
                ]
            ),
        )

    gm.press(_graph, parents=[4, 5])
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['xyposition'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    with unfiltered(_graph):
        assert np.allclose(
            _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
            np.array(
                [
                    [
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        1.85,
                        2.0,
                        2.15,
                        2.05,
                        2.25,
                        4.66666667,
                    ],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.15, 0.0, 0.15, 0.05, 0.25, 0.0],
                ]
            ),
        )

    gm.press(_graph, parents=[3, 11])
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['xyposition'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    with unfiltered(_graph):
        assert np.allclose(
            _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
            np.array(
                [
                    [
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        1.85,
                        2.0,
                        2.15,
                        2.05,
                        2.25,
                        4.66666667,
                        4.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -0.15,
                        0.0,
                        0.15,
                        0.05,
                        0.25,
                        0.0,
                        0.0,
                    ],
                ]
            ),
        )


def test_graph_depth():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    num_vertices = _graph.num_vertices(ignore_filter=True)
    # TODO: Assign depths to _depth
    _length = _graph.new_vertex_property("int", vals=[1, 2, 1, 2, 1, 2])
    _sequence = _graph.new_vertex_property("string", vals=range(len(_graph)))

    depth = np.array(
        [
            [0, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ]
    )
    _depth = _graph.new_vertex_property("vector<float>")
    _depth.set_2d_array(depth, pos=[0, 1])

    offset_scale = 0.1
    xyposition = np.empty((2, num_vertices))
    xyposition[0, :] = np.arange(num_vertices)
    xyposition[1, :] = 0
    _xyposition = _graph.new_vertex_property("vector<float>")
    _xyposition.set_2d_array(xyposition, pos=[0, 1])

    _filter = _graph.new_vertex_property("bool", val=1)

    _graph.vp["depth"] = _depth
    _graph.vp["length"] = _length
    _graph.vp["sequence"] = _sequence
    _graph.vp["xyposition"] = _xyposition
    _graph.vp["filter"] = _filter

    _graph.set_vertex_filter(_graph.vp["filter"])

    gm = sz.graph_manager.GraphManager(
        unzippers=[
            sz.graph_manager.LengthUnzipper(),
            sz.graph_manager.SequenceUnzipper(),
            sz.graph_manager.VectorDepthUnzipper(),
            sz.graph_manager.PositionUnzipper(offset=(0.1, 0.1)),
        ],
        pressers=[
            sz.graph_manager.LengthPresser(),
            sz.graph_manager.SequencePresser(sep=","),
            sz.graph_manager.VectorDepthPresser(),
            sz.graph_manager.PositionPresser(),
        ],
    )
    gm.validate(_graph)

    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['depth'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    with unfiltered(_graph):
        assert np.allclose(
            _graph.vp["depth"].get_2d_array(pos=[0, 1]),
            np.array([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        )

    gm.unzip(
        _graph,
        2,
        [(1, 3), (1, 3), (1, 3)],
        path_depths=[[0.33, 0], [0.33, 0], [0.33, 0.2]],
    )
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['depth'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    with unfiltered(_graph):
        assert np.allclose(
            _graph.vp["depth"].get_2d_array(pos=[0, 1]),
            np.array(
                [
                    [0.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.33, 0.33, 0.33],
                    [1.0, 1.0, -0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2],
                ]
            ),
        )

    gm.unzip(_graph, 8, [(1, 3), (1, 3)], path_depths=[[0.3, 0], [0.1, 0]])
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['depth'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    with unfiltered(_graph):
        assert np.allclose(
            _graph.vp["depth"].get_2d_array(pos=[0, 1]),
            np.array(
                [
                    [0.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.33, 0.33, -0.07, 0.3, 0.1],
                    [1.0, 1.0, -0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0, 0.0],
                ]
            ),
        )

    gm.press(_graph, parents=[4, 5])
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['depth'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    with unfiltered(_graph):
        assert np.allclose(
            _graph.vp["depth"].get_2d_array(pos=[0, 1]),
            np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.01,
                        0.0,
                        0.66666667,
                        -0.33333333,
                        0.33,
                        0.33,
                        -0.07,
                        0.3,
                        0.1,
                        0.33333333,
                    ],
                    [
                        1.0,
                        1.0,
                        -0.2,
                        0.0,
                        -0.66666667,
                        0.33333333,
                        0.0,
                        0.0,
                        0.2,
                        0.0,
                        0.0,
                        0.66666667,
                    ],
                ]
            ),
        )

    gm.press(_graph, parents=[3, 11])
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['depth'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    with unfiltered(_graph):
        assert np.allclose(
            _graph.vp["depth"].get_2d_array(pos=[0, 1]),
            np.array(
                [
                    [
                        0.0,
                        0.0,
                        0.01,
                        -0.2,
                        0.66666667,
                        -0.33333333,
                        0.33,
                        0.33,
                        -0.07,
                        0.3,
                        0.1,
                        0.13333333,
                        0.2,
                    ],
                    [
                        1.0,
                        1.0,
                        -0.2,
                        -0.4,
                        -0.66666667,
                        0.33333333,
                        0.0,
                        0.0,
                        0.2,
                        0.0,
                        0.0,
                        0.26666667,
                        0.4,
                    ],
                ]
            ),
        )


def test_unzip_topology():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    # gt.draw.graph_draw(gt.GraphView(_graph), ink_scale=0.35, vertex_text=_graph.vertex_index)

    _graph.vp["filter"] = _graph.new_vertex_property("bool", val=True)
    _graph.set_vertex_filter(_graph.vp["filter"])

    gm = sz.graph_manager.GraphManager()
    gm.validate(_graph)

    gm.unzip(_graph, 3, [(2, 4), (2, 4)])
    gm.unzip(_graph, 5, [(4, 6), (4, 6)])

    # sz.draw.draw_graph(_graph)
    # print(repr(sz.stats.degree_stats(_graph).reset_index().values))
    assert np.array_equal(
        sz.stats.degree_stats(_graph).reset_index().values,
        [
            [1.0, 1.0, 5.0],
            [0.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 2.0, 1.0],
        ],
    )

    gm.unzip(_graph, 4, [(7, 9), (7, 10), (8, 9), (8, 10)])

    # sz.draw.draw_graph(_graph)
    # print(repr(sz.stats.degree_stats(_graph).reset_index().values))
    assert np.array_equal(
        sz.stats.degree_stats(_graph).reset_index().values,
        [
            [1.0, 1.0, 5.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 2.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
    )


def test_batch_unzip_topology_simple():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    # gt.draw.graph_draw(gt.GraphView(_graph), ink_scale=0.35, vertex_text=_graph.vertex_index)

    _graph.vp["filter"] = _graph.new_vertex_property("bool", val=True)
    _graph.set_vertex_filter(_graph.vp["filter"])

    gm = sz.graph_manager.GraphManager()
    gm.validate(_graph)

    gm.batch_unzip(
        _graph,
        (3, [(2, 4), (2, 4)], {}),
        (5, [(4, 6), (4, 6)], {}),
    )
    # Should be equivalent to:
    # gm.unzip(_graph, 3, [(2, 4), (2, 4)])
    # gm.unzip(_graph, 5, [(4, 6), (4, 6)])
    # Because splitting the middle node (4) now has 2*2 paths that it can traverse.

    # sz.draw.draw_graph(_graph)
    # print(repr(sz.stats.degree_stats(_graph).reset_index().values))
    assert np.array_equal(
        sz.stats.degree_stats(_graph).reset_index().values,
        [
            [1.0, 1.0, 5.0],
            [0.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 2.0, 1.0],
        ],
    )


def test_batch_unzip_topology_complex1():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (3, 1), (2, 4), (2, 5)])

    _graph.vp["filter"] = _graph.new_vertex_property("bool", val=True)
    _graph.set_vertex_filter(_graph.vp["filter"])

    gm = sz.graph_manager.GraphManager()
    gm.validate(_graph)

    gm.batch_unzip(
        _graph,
        (2, [(1, 5), (1, 4)], {}),
        (1, [(0, 2), (3, 2)], {}),
    )

    assert np.array_equal(
        sz.stats.degree_stats(_graph).reset_index().values,
        [[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [1.0, 2.0, 2.0], [2.0, 1.0, 2.0]],
    )


def test_batch_unzip_topology_complex2():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    # gt.draw.graph_draw(gt.GraphView(_graph), ink_scale=0.35, vertex_text=_graph.vertex_index)

    _graph.vp["filter"] = _graph.new_vertex_property("bool", val=True)
    _graph.set_vertex_filter(_graph.vp["filter"])

    gm = sz.graph_manager.GraphManager()
    gm.validate(_graph)

    gm.batch_unzip(
        _graph,
        (3, [(2, 4), (2, 4)], {}),
        (5, [(4, 6), (4, 6)], {}),
        (4, [(3, 5), (3, 5)], {}),
    )

    # sz.draw.draw_graph(_graph)
    # print(repr(sz.stats.degree_stats(_graph).reset_index().values))
    assert np.array_equal(
        sz.stats.degree_stats(_graph).reset_index().values,
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 2.0],
            [2.0, 2.0, 2.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
    )


def test_batch_operations_on_properties():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    num_vertices = _graph.num_vertices(ignore_filter=True)

    _length = _graph.new_vertex_property("int", val=1)
    _sequence = _graph.new_vertex_property("string")
    _depth = _graph.new_vertex_property("float")
    _xyposition = _graph.new_vertex_property("vector<float>")
    _filter = _graph.new_vertex_property("bool", val=1)

    # Initialize position info
    xyposition = np.empty((2, num_vertices))
    xyposition[0, :] = np.arange(num_vertices)
    xyposition[1, :] = 0
    _xyposition = _graph.new_vertex_property("vector<float>")
    _xyposition.set_2d_array(xyposition, pos=[0, 1])

    _graph.vp["depth"] = _depth
    _graph.vp["length"] = _length
    _graph.vp["sequence"] = _sequence
    _graph.vp["xyposition"] = _xyposition
    _graph.vp["filter"] = _filter

    _graph.set_vertex_filter(_graph.vp["filter"])

    gm = sz.graph_manager.GraphManager(
        unzippers=[
            sz.graph_manager.LengthUnzipper(),
            sz.graph_manager.SequenceUnzipper(),
            sz.graph_manager.ScalarDepthUnzipper(),
            sz.graph_manager.PositionUnzipper(offset=(0.1, 0.1)),
        ],
        pressers=[
            sz.graph_manager.LengthPresser(),
            sz.graph_manager.SequencePresser(sep=","),
            sz.graph_manager.ScalarDepthPresser(),
            sz.graph_manager.PositionPresser(),
        ],
    )
    gm.validate(_graph)

    gm.batch_unzip(
        _graph,
        (3, [(2, 4), (2, 4)], {"path_depths": [0, 0]}),
        (4, [(3, 5), (3, 5)], {"path_depths": [0, 0]}),
        (5, [(4, 6), (4, 6)], {"path_depths": [0, 0]}),
    )

    with unfiltered(_graph):
        assert np.array_equal(
            _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.9, 3.1, 3.9, 4.1, 4.9, 5.1],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.1,
                    0.1,
                    -0.1,
                    0.1,
                    -0.1,
                    0.1,
                ],
            ],
        )

    gm.batch_press(
        _graph,
        ([0, 1, 2], {}),
        ([6, 7], {}),
    )
    with unfiltered(_graph):
        assert np.array_equal(
            _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
            np.array(
                [
                    [
                        0.0,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        2.9,
                        3.1,
                        3.9,
                        4.1,
                        4.9,
                        5.1,
                        1.0,
                        6.5,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -0.1,
                        0.1,
                        -0.1,
                        0.1,
                        -0.1,
                        0.1,
                        0.0,
                        0.0,
                    ],
                ]
            ),
        )
