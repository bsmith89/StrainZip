from typing import cast

import graph_tool as gt
import graph_tool.draw
import numpy as np

import strainzip as sz


def test_viz_graph_positioning():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    num_vertices = _graph.num_vertices(ignore_filter=True)
    _depth = _graph.new_vertex_property("float")
    # TODO: Assign depths to _depth
    _length = _graph.new_vertex_property("int", vals=[1, 2, 1, 2, 1, 2])
    _sequence = _graph.new_vertex_property("string", vals=range(len(_graph)))

    offset_scale = 0.1
    xyposition = np.empty((2, num_vertices))
    xyposition[0, :] = np.arange(num_vertices)
    xyposition[1, :] = 0
    _position = _graph.new_vertex_property("vector<float>")
    _position.set_2d_array(xyposition, pos=[0, 1])
    xyposition = _position

    g = sz.VizZipGraph(
        graph=_graph,
        depth=sz.DepthProperty(_depth),
        length=sz.LengthProperty(_length),
        sequence=sz.SequenceProperty(_sequence),
        xyposition=sz.PositionProperty(xyposition),
        pos_offset_scale=offset_scale,
        # init_step = 0 means that the sfdp_layout update steps does nothing to the positions.
        sfdp_layout_kwargs=dict(init_step=0),
    )

    # gt.draw.graph_draw(g.graph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index)
    assert np.allclose(
        g.props["xyposition"].vprop.get_2d_array(pos=[0, 1]),
        np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )

    g.unzip(2, [(1, 3), (1, 3), (1, 3)], **{"depth": [0, 0, 0]})
    # gt.draw.graph_draw(g.graph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index)
    assert np.allclose(
        g.props["xyposition"].vprop.get_2d_array(pos=[0, 1]),
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.9, 2.0, 2.1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.1],
            ]
        ),
    )

    g.unzip(8, [(1, 3), (1, 3)], **{"depth": [0, 0]})
    # gt.draw.graph_draw(g.graph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index)
    assert np.allclose(
        g.props["xyposition"].vprop.get_2d_array(pos=[0, 1]),
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.9, 2.0, 2.1, 2.0, 2.2],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.0, 0.2],
            ]
        ),
    )

    g.press(parents=[4, 5])
    # gt.draw.graph_draw(g.graph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index)
    assert np.allclose(
        g.props["xyposition"].vprop.get_2d_array(pos=[0, 1]),
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.9, 2.0, 2.1, 2.0, 2.2, 4.66666667],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.0, 0.2, 0.0],
            ]
        ),
    )

    g.press(parents=[3, 11])
    # gt.draw.graph_draw(g.graph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index)
    assert np.allclose(
        g.props["xyposition"].vprop.get_2d_array(pos=[0, 1]),
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

    g.press(parents=[0, 1])
    # gt.draw.graph_draw(g.graph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index)
    assert np.allclose(
        g.props["xyposition"].vprop.get_2d_array(pos=[0, 1]),
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
                    0.66666667,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_viz_graph_depths():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    num_vertices = _graph.num_vertices(ignore_filter=True)
    # TODO: Assign depths to _depth
    _length = _graph.new_vertex_property("int", vals=[1, 2, 1, 2, 1, 2])
    _sequence = _graph.new_vertex_property("string", vals=range(len(_graph)))

    depth = np.array([[0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1]])
    _depth = _graph.new_vertex_property("vector<float>")
    _depth.set_2d_array(depth, pos=[0, 1])

    offset_scale = 0.1
    xyposition = np.empty((2, num_vertices))
    xyposition[0, :] = np.arange(num_vertices)
    xyposition[1, :] = 0
    _position = _graph.new_vertex_property("vector<float>")
    _position.set_2d_array(xyposition, pos=[0, 1])
    xyposition = _position

    g = sz.VizZipGraph(
        graph=_graph,
        depth=sz.DepthProperty(_depth),
        length=sz.LengthProperty(_length),
        sequence=sz.SequenceProperty(_sequence),
        xyposition=sz.PositionProperty(xyposition),
        pos_offset_scale=offset_scale,
        # init_step = 0 means that the sfdp_layout update steps does nothing to the positions.
        sfdp_layout_kwargs=dict(init_step=0),
    )

    # print(repr(g.props['depth'].vprop.get_2d_array(pos=[0, 1])))
    assert np.allclose(
        g.props["depth"].vprop.get_2d_array(pos=[0, 1]),
        np.array([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
    )
    # gt.draw.graph_draw(g.fgraph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index, vertex_color=gt.ungroup_vector_property(g.props['depth'].vprop, pos=[0])[0])

    g.unzip(
        2, [(1, 3), (1, 3), (1, 3)], **{"depth": [[0.33, 0], [0.33, 0], [0.33, 0.2]]}
    )
    # print(repr(g.props['depth'].vprop.get_2d_array(pos=[0, 1])))
    assert np.allclose(
        g.props["depth"].vprop.get_2d_array(pos=[0, 1]),
        np.array(
            [
                [0.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.33, 0.33, 0.33],
                [1.0, 1.0, -0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2],
            ]
        ),
    )
    # gt.draw.graph_draw(g.fgraph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index, vertex_color=gt.ungroup_vector_property(g.props['depth'].vprop, pos=[0])[0])

    g.unzip(8, [(1, 3), (1, 3)], **{"depth": [[0.3, 0], [0.1, 0]]})
    # print(repr(g.props['depth'].vprop.get_2d_array(pos=[0, 1])))
    assert np.allclose(
        g.props["depth"].vprop.get_2d_array(pos=[0, 1]),
        np.array(
            [
                [0.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.33, 0.33, -0.07, 0.3, 0.1],
                [1.0, 1.0, -0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 0.0, 0.0],
            ]
        ),
    )
    # gt.draw.graph_draw(g.fgraph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index, vertex_color=gt.ungroup_vector_property(g.props['depth'].vprop, pos=[0])[0])

    g.press(parents=[4, 5])
    # print(repr(g.props['depth'].vprop.get_2d_array(pos=[0, 1])))
    assert np.allclose(
        g.props["depth"].vprop.get_2d_array(pos=[0, 1]),
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
    # gt.draw.graph_draw(g.fgraph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index, vertex_color=gt.ungroup_vector_property(g.props['depth'].vprop, pos=[0])[0])

    g.press(parents=[3, 11])
    # print(repr(g.props['depth'].vprop.get_2d_array(pos=[0, 1])))
    assert np.allclose(
        g.props["depth"].vprop.get_2d_array(pos=[0, 1]),
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
    # gt.draw.graph_draw(g.fgraph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index, vertex_color=gt.ungroup_vector_property(g.props['depth'].vprop, pos=[0])[0])

    g.press(parents=[0, 1])
    # print(repr(g.props['depth'].vprop.get_2d_array(pos=[0, 1])))
    assert np.allclose(
        g.props["depth"].vprop.get_2d_array(pos=[0, 1]),
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
                    0.0,
                ],
                [
                    0.0,
                    0.0,
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
                    1.0,
                ],
            ]
        ),
    )
    # gt.draw.graph_draw(g.fgraph, pos=g.props['xyposition'].vprop, ink_scale=0.35, vertex_text=g.graph.vertex_index, vertex_color=gt.ungroup_vector_property(g.props['depth'].vprop, pos=[0])[0])
