from warnings import warn

import graph_tool as gt
import numpy as np


def estimate_flow(graph, depth, weight, eps=0.001, maxiter=1000):
    target_vertex_weight = gt.edge_endpoint_property(graph, weight, "target")
    source_vertex_weight = gt.edge_endpoint_property(graph, weight, "source")
    flow = graph.new_edge_property("float", val=1)
    loss_hist = [np.finfo("float").max]
    for _ in range(maxiter):
        # Inflow error
        total_in_flow = gt.incident_edges_op(graph, "in", "sum", flow)
        in_flow_error = graph.new_vertex_property(
            "float", vals=depth.a - total_in_flow.a
        )
        target_vertex_total_inflow = gt.edge_endpoint_property(
            graph, total_in_flow, "target"
        )
        target_vertex_error = gt.edge_endpoint_property(graph, in_flow_error, "target")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            target_vertex_alloc = np.nan_to_num(
                flow.a / target_vertex_total_inflow.a, posinf=1, nan=0
            )
        target_vertex_alloc_error = target_vertex_alloc * target_vertex_error.a

        # Outflow error
        total_out_flow = gt.incident_edges_op(graph, "out", "sum", flow)
        out_flow_error = graph.new_vertex_property(
            "float", vals=depth.a - total_out_flow.a
        )
        source_vertex_total_outflow = gt.edge_endpoint_property(
            graph, total_out_flow, "source"
        )
        source_vertex_error = gt.edge_endpoint_property(graph, out_flow_error, "source")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            source_vertex_alloc = np.nan_to_num(
                flow.a / source_vertex_total_outflow.a, posinf=1, nan=0
            )
        source_vertex_alloc_error = source_vertex_alloc * source_vertex_error.a

        # Loss <- Sum of squared in/out-flow errors
        loss_hist.append(
            np.square(in_flow_error.a).sum() + np.square(out_flow_error.a).sum()
        )
        if loss_hist[-1] == 0:
            break  # This should only happen if d is all 0's.
        loss_ratio = (loss_hist[-2] - loss_hist[-1]) / loss_hist[-2]
        if loss_ratio < eps:
            break

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            # NOTE: Some values of (source_vertex_weight.a + target_vertex_weight.a)
            # are 0 because these two edge_properties include edge indices
            # for non-existent edges.
            # TODO: Consider running gt.reindex_edges to get rid of these.
            mean_flow_error = graph.new_edge_property(
                "float",
                vals=(
                    (source_vertex_alloc_error * source_vertex_weight.a)
                    + (target_vertex_alloc_error * target_vertex_weight.a)
                )
                / (source_vertex_weight.a + target_vertex_weight.a),
            )
        flow = graph.new_edge_property("float", vals=flow.a + mean_flow_error.a)
    else:
        warn("Reached maxiter. Flow estimates did not converge.")
    return flow
