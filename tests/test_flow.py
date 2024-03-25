import graph_tool as gt

import strainzip as sz


def test_simple_flow():
    graph = gt.Graph()
    graph.add_edge_list([(0, 2), (1, 2), (2, 3), (2, 4)])

    length = graph.new_vertex_property("int", val=1)

    depth = graph.new_vertex_property("float", val=1)
    depth.a[0] = 10
    depth.a[2] = 5
    depth.a[3] = 2

    flow = sz.flow.estimate_flow(
        graph, depth=depth, weight=length, eps=0.001, maxiter=1000
    )
