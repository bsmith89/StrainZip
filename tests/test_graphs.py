from typing import cast

import graph_tool as gt
import graph_tool.draw
import numpy as np

import strainzip as sz


def test_viz_graph():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4)])
    _depth = _graph.new_vertex_property("float")
    # TODO: Assign depths to _depth
    _length = _graph.new_vertex_property("int", vals=[1, 2, 1, 2, 1])
    _sequence = _graph.new_vertex_property("string", vals=range(len(_graph)))

    np.random.seed(2)
    gt.seed_rng(2)
    offset_scale = 0.1
    _xcoord = _graph.new_vertex_property("float", vals=np.linspace(0, 1, num=5))
    _ycoord = _graph.new_vertex_property(
        "float", vals=np.random.uniform(-offset_scale, offset_scale, size=5)
    )

    g = sz.VizZipGraph(
        graph=_graph,
        depth=sz.DepthProperty(_depth),
        length=sz.LengthProperty(_length),
        sequence=sz.SequenceProperty(_sequence),
        xcoord=sz.CoordinateProperty(_xcoord),
        ycoord=sz.CoordinateProperty(_ycoord),
        coord_offset_scale=offset_scale,
    )

    g.unzip(2, [(1, 3), (1, 3)], **{"depth": [0, 0]})
    g.unzip(5, [(1, 3), (1, 3)], **{"depth": [0, 0]})
    g.press(parents=[3, 4])
    g.press(parents=[0, 1])
