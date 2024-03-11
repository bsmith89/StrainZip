from typing import cast

import graph_tool as gt
import graph_tool.draw as gtdraw

from strainzip.graph import DepthGraph
from strainzip.vertex_properties import (
    CoordinateProperty,
    DepthProperty,
    LengthProperty,
    SequenceProperty,
)


def test_graph_initialization():
    _graph = gt.Graph()
    _graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 4)])
    _depth = _graph.new_vertex_property("float")
    # TODO: Assign depths to _depth
    _length = _graph.new_vertex_property("int", val=1)
    _sequence = _graph.new_vertex_property("string", vals=range(len(_graph)))
    _xcoordinate, _ycoordinate = gt.ungroup_vector_property(
        gtdraw.sfdp_layout(_graph), pos=[0, 1]
    )
    g = DepthGraph(
        graph=_graph,
        depth=DepthProperty(_depth),
        length=LengthProperty(_length),
        sequence=SequenceProperty(_sequence),
        xcoordinate=CoordinateProperty(_xcoordinate),
        ycoordinate=CoordinateProperty(_ycoordinate),
    )
