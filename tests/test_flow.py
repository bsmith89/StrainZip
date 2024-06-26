import graph_tool as gt
import numpy as np
import pytest

import strainzip as sz


def test_estimate_flow_convergence_warning():
    graph = gt.Graph()
    graph.add_edge_list([(0, 2), (1, 2), (2, 3), (2, 4)])

    length = graph.new_vertex_property("int", val=1)

    depth = graph.new_vertex_property("float", val=1)
    depth.a[0] = 10
    depth.a[2] = 5
    depth.a[3] = 2

    with pytest.warns(UserWarning):
        flow, hist = sz.flow.estimate_flow(
            graph, depth=depth, length=length, eps=1e-10, maxiter=10
        )


def test_estimate_flow_cycle():
    graph = gt.Graph()
    graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 0)])

    length = graph.new_vertex_property("int", val=1)

    depth = graph.new_vertex_property("float", val=1)
    depth.a[0] = 2

    flow, hist = sz.flow.estimate_flow(
        graph, depth=depth, length=length, eps=1e-10, maxiter=1000
    )
    assert flow[(0, 1)] == 1.5
    assert flow[(1, 2)] == 1.0
    assert flow[(2, 3)] == 1.0
    assert flow[(3, 0)] == 1.5


def test_estimate_flow_minor_depth():
    graph = gt.Graph()
    graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)])

    length = graph.new_vertex_property("int", val=1)

    depth = graph.new_vertex_property("float", val=1)
    depth.a[4] = 1e-2

    flow, hist = sz.flow.estimate_flow(
        graph, depth=depth, length=length, eps=1e-10, maxiter=1000
    )
    assert flow[(0, 1)] < 1.0
    assert (
        flow[(0, 4)] < 1e-2
    ), "Since (0, 4) is the stem of the 4-lolipop, it should only lose depth."


def test_estimate_flow_filtered_graph():
    graph = gt.Graph()
    graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)])
    filter = graph.new_vertex_property("bool", val=1)
    graph.set_vertex_filter(filter)

    length = graph.new_vertex_property("int", val=1)

    depth = graph.new_vertex_property("float", val=1)
    depth.a[4] = 1

    flow, hist = sz.flow.estimate_flow(
        graph, depth=depth, length=length, eps=1e-10, maxiter=1000
    )
    assert flow[(0, 1)] < 1.0
    assert flow[(0, 4)] < 1.0

    filter.a[4] = 0
    flow, hist = sz.flow.estimate_flow(
        graph, depth=depth, length=length, eps=1e-10, maxiter=1000
    )
    assert flow[(0, 1)] == 1.0


def test_flow_on_big_graph_with_tips():
    np.random.seed(1)
    gt.seed_rng(1)
    sequence = sz.sequence.random_sequence(1000)
    graph = sz.build.annotated_dbg(sequence, k=7, circularize=False, include_rc=True)
    flow, hist = sz.flow.estimate_flow(
        graph,
        depth=graph.vp["depth"],
        length=graph.vp["length"],
        eps=1e-10,
        maxiter=1000,
    )


def test_estimate_flow_with_blunt_tips():
    graph = gt.Graph([(0, 1), (1, 2), (2, 3), (1, 4), (4, 1)])
    depth = graph.new_vertex_property("float", vals=[1, 1, 1, 1, 2])
    length = graph.new_vertex_property("float", vals=[1, 1, 1, 1, 1])
    graph.vp["filter"] = graph.new_vertex_property("bool", val=1)
    graph.set_vertex_filter(graph.vp["filter"])
    flow, hist = sz.flow.estimate_flow(
        graph, depth=depth, length=length, eps=1e-10, maxiter=1000
    )
    # TODO: Check that flow estimates make sense.


def test_estimate_flow_with_filtered_vertices():
    graph = gt.Graph([(0, 1), (1, 2), (2, 3), (1, 4), (4, 1)])
    depth = graph.new_vertex_property("float", vals=[1, 1, 1, 1, 0])
    length = graph.new_vertex_property("float", vals=[1, 1, 1, 1, 0])
    graph.vp["filter"] = graph.new_vertex_property("bool", val=1)
    graph.set_vertex_filter(graph.vp["filter"])
    graph.vp["filter"].a[4] = 0  # Filter vertex 4

    flow, hist = sz.flow.estimate_flow(
        graph, depth=depth, length=length, eps=1e-10, maxiter=1000
    )
    # TODO: Check that flow estimates make sense.
