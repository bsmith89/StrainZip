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
            graph, depth=depth, length=length, eps=0.001, maxiter=3
        )


def test_estimate_flow_cycle():
    graph = gt.Graph()
    graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 0)])

    length = graph.new_vertex_property("int", val=1)

    depth = graph.new_vertex_property("float", val=1)
    depth.a[0] = 2

    flow, hist = sz.flow.estimate_flow(
        graph, depth=depth, length=length, eps=0.001, maxiter=1000
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
    depth.a[4] = 1e-5

    flow, hist = sz.flow.estimate_flow(
        graph, depth=depth, length=length, eps=0.001, maxiter=1000
    )
    assert flow[(0, 1)] < 1.0
    assert flow[(0, 4)] > 1e-5


def test_estimate_flow_filtered_graph():
    graph = gt.Graph()
    graph.add_edge_list([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)])
    filter = graph.new_vertex_property("bool", val=1)
    graph.set_vertex_filter(filter)

    length = graph.new_vertex_property("int", val=1)

    depth = graph.new_vertex_property("float", val=1)
    depth.a[4] = 1

    flow, hist = sz.flow.estimate_flow(
        graph, depth=depth, weight=length, eps=0.001, maxiter=1000
    )
    assert flow[(0, 1)] < 1.0
    assert flow[(0, 4)] < 1.0

    filter.a[4] = 0
    flow, hist = sz.flow.estimate_flow(
        graph, depth=depth, weight=length, eps=0.001, maxiter=1000
    )
    assert flow[(0, 1)] == 1.0


def test_flow_on_graph_with_tips():
    np.random.seed(1)
    gt.seed_rng(1)
    sequence = sz.sequence.random_sequence(1000)
    graph = sz.build.annotated_dbg(sequence, k=7, circularize=False, include_rc=True)
    flow, hist = sz.flow.estimate_flow(
        graph,
        depth=graph.vp["depth"],
        length=graph.vp["length"],
        eps=0.001,
        maxiter=1000,
    )
