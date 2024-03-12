from typing import cast

import graph_tool as gt
import graph_tool.draw
import numpy as np

import strainzip as sz


def test_viz_graph_positioning():
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

    gm = sz.graph.PositionGraphManager(
        pos_offset_scale=offset_scale,
        # init_step = 0 means that the sfdp_layout update steps does nothing to the positions.
        sfdp_layout_kwargs=dict(init_step=0.0, max_iter=1),
    )
    gm.validate_graph(_graph)
    gm.validate_manager(_graph)

    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['xyposition'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
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
    assert np.allclose(
        _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.9, 2.0, 2.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.1],
            ]
        ),
    )

    gm.unzip(_graph, 8, [(1, 3), (1, 3)], path_depths=[[0.3, 0], [0.1, 0]])
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['xyposition'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    assert np.allclose(
        _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.9, 2.0, 2.1, 2.0, 2.2],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.0, 0.2],
            ]
        ),
    )

    gm.press(_graph, parents=[4, 5])
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['xyposition'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    assert np.allclose(
        _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.9, 2.0, 2.1, 2.0, 2.2, 4.66666667],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.0, 0.2, 0.0],
            ]
        ),
    )

    gm.press(_graph, parents=[3, 11])
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['xyposition'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
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
                    1.9,
                    2.0,
                    2.1,
                    2.0,
                    2.2,
                    4.66666667,
                    4.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0],
            ]
        ),
    )


def test_viz_graph_depth():
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

    gm = sz.graph.PositionGraphManager(
        pos_offset_scale=offset_scale,
        # init_step = 0 means that the sfdp_layout update steps does nothing to the positions.
        sfdp_layout_kwargs=dict(init_step=0.0, max_iter=1),
    )
    gm.validate_graph(_graph)
    gm.validate_manager(_graph)

    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['depth'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
    # print(repr(_graph.vp['depth'].get_2d_array(pos=[0, 1])))
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


def test_batch_operations_topology():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    # gt.draw.graph_draw(gt.GraphView(_graph), ink_scale=0.35, vertex_text=_graph.vertex_index)

    gm = sz.graph.BaseGraphManager()
    gm.validate_graph(_graph)
    gm.validate_manager(_graph)

    gm.batch_unzip(_graph, *[(3, [(2, 4)], {})] * 2)
    # gm.unzip(_graph, 3, [(2, 4)])
    # gm.unzip(_graph, 3, [(2, 4)]) # Should be equivalent to the above

    assert np.array_equal(_graph.get_in_degrees([2, 4, 7, 8]), [1, 3, 1, 1])
    assert np.array_equal(_graph.get_out_degrees([2, 4, 7, 8]), [3, 1, 1, 1])
    # gt.draw.graph_draw(gt.GraphView(_graph), ink_scale=0.35, vertex_text=_graph.vertex_index)

    gm.batch_press(_graph, *[([0, 1, 2], {}), ([4, 5, 6], {})])
    # gm.press(_graph, [0, 1, 2])
    # gm.press(_graph, [4, 5, 6]) # Should be equivalent to the above

    assert np.array_equal(_graph.get_in_degrees([9, 10]), [0, 3])
    assert np.array_equal(_graph.get_out_degrees([9, 10]), [3, 0])
    # gt.draw.graph_draw(gt.GraphView(_graph), ink_scale=0.35, vertex_text=_graph.vertex_index)


def test_batch_operations_on_position_graph():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    num_vertices = _graph.num_vertices(ignore_filter=True)

    _length = _graph.new_vertex_property("int", val=1)
    _sequence = _graph.new_vertex_property("string")
    _depth = _graph.new_vertex_property("float")
    _xyposition = _graph.new_vertex_property("vector<float>")
    _filter = _graph.new_vertex_property("bool", val=1)

    # Initialize position info
    offset_scale = 0.1
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

    gm = sz.graph.PositionGraphManager(sfdp_layout_kwargs=dict(init_step=0.0))
    gm.validate_graph(_graph)
    gm.validate_manager(_graph)
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(_graph.vp['xyposition'].get_2d_array(pos=[0, 1]))

    gm.batch_unzip(_graph, *[(3, [(2, 4)], {"path_depths": [0]})] * 2)
    # gm.unzip(_graph, 3, [(2, 4)], path_depths=[0])
    # gm.unzip(_graph, 3, [(2, 4)], path_depths=[0]) # Should be equivalent to the above
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(_graph.vp['xyposition'].get_2d_array(pos=[0, 1]))

    gm.batch_press(_graph, *[([0, 1, 2], {}), ([4, 5, 6], {})])
    # gm.press(_graph, [0, 1, 2])
    # gm.press(_graph, [4, 5, 6]) # Should be equivalent to the above
    # gt.draw.graph_draw(gt.GraphView(_graph, vfilt=_graph.vp['filter']), pos=_graph.vp['xyposition'], ink_scale=0.35, vertex_text=_graph.vertex_index)
    # print(repr(_graph.vp['xyposition'].get_2d_array(pos=[0, 1])))
    assert np.array_equal(
        _graph.vp["xyposition"].get_2d_array(pos=[0, 1]),
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.9, 2.9, 1.0, 5.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.1, 0.0, 0.0],
            ]
        ),
    )
