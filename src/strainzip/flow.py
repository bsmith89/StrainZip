from warnings import warn

import graph_tool as gt
import numpy as np
from tqdm import tqdm


def estimate_flow(graph, depth, weight, eps=0.001, maxiter=1000, verbose=False):
    target_vertex_weight = gt.edge_endpoint_property(graph, weight, "target")
    source_vertex_weight = gt.edge_endpoint_property(graph, weight, "source")

    # Allocate PropertyMaps
    flow = graph.new_edge_property("float", val=1)
    # In / Target
    total_in_flow = graph.new_vertex_property("float")
    in_flow_error = graph.new_vertex_property("float")
    target_vertex_total_inflow = graph.new_edge_property("float")
    target_vertex_error = graph.new_edge_property("float")
    target_vertex_alloc = np.empty_like(flow.a)
    target_vertex_alloc_error = np.empty_like(flow.a)
    # Out / Source
    total_out_flow = graph.new_vertex_property("float")
    out_flow_error = graph.new_vertex_property("float")
    source_vertex_total_outflow = graph.new_edge_property("float")
    source_vertex_error = graph.new_edge_property("float")
    source_vertex_alloc = np.empty_like(flow.a)
    source_vertex_alloc_error = np.empty_like(flow.a)

    loss_hist = [np.finfo("float").max]
    i = 0
    pbar = tqdm(
        range(maxiter),
        total=maxiter,
        mininterval=1.0,
        bar_format="{l_bar}{r_bar}",
        disable=(not verbose),
    )
    for i in pbar:
        # Update inflow error
        gt.incident_edges_op(graph, "in", "sum", flow, total_in_flow)
        np.subtract(depth.a, total_in_flow.a, out=in_flow_error.a)
        gt.edge_endpoint_property(
            graph, total_in_flow, "target", target_vertex_total_inflow
        )
        # Make sure that we don't get a NaN in the next step?
        target_vertex_total_inflow.a[:] = np.where(
            target_vertex_total_inflow.a == 0, np.inf, target_vertex_total_inflow.a
        )
        gt.edge_endpoint_property(graph, in_flow_error, "target", target_vertex_error)
        np.divide(flow.a, target_vertex_total_inflow.a, out=target_vertex_alloc)
        target_vertex_alloc = np.nan_to_num(
            flow.a / target_vertex_total_inflow.a, posinf=1, nan=0, copy=False
        )
        np.multiply(
            target_vertex_alloc, target_vertex_error.a, out=target_vertex_alloc_error
        )

        # Outflow error
        gt.incident_edges_op(graph, "out", "sum", flow, total_out_flow)
        np.subtract(depth.a, total_out_flow.a, out=out_flow_error.a)
        gt.edge_endpoint_property(
            graph, total_out_flow, "source", source_vertex_total_outflow
        )
        source_vertex_total_outflow.a[:] = np.where(
            source_vertex_total_outflow.a == 0, np.inf, source_vertex_total_outflow.a
        )
        gt.edge_endpoint_property(graph, out_flow_error, "source", source_vertex_error)
        np.divide(flow.a, source_vertex_total_outflow.a, out=source_vertex_alloc)
        source_vertex_alloc = np.nan_to_num(
            flow.a / source_vertex_total_outflow.a, posinf=1, nan=0, copy=False
        )
        np.multiply(
            source_vertex_alloc, source_vertex_error.a, out=source_vertex_alloc_error
        )

        # Loss <- Sum of squared in/out-flow errors
        loss_hist.append(
            float(np.square(in_flow_error.a).sum() + np.square(out_flow_error.a).sum())  # type: ignore[reportArgumentType]
        )
        if loss_hist[-1] == 0:
            break  # This should only happen if d is all 0's.
        loss_ratio = (loss_hist[-2] - loss_hist[-1]) / loss_hist[-2]
        pbar.set_postfix({"improvement": loss_ratio})
        if loss_ratio < eps:
            break

        # NOTE: Some values of (source_vertex_weight.a + target_vertex_weight.a)
        # are 0 because these two edge_properties include edge indices
        # for non-existent edges.
        # TODO: Consider running gt.reindex_edges to get rid of these.
        mean_flow_error = (
            (source_vertex_alloc_error * source_vertex_weight.a)
            + (target_vertex_alloc_error * target_vertex_weight.a)
        ) / (source_vertex_weight.a + target_vertex_weight.a)
        flow.a += mean_flow_error
    else:
        warn("Reached maxiter. Flow estimates did not converge.")

    # Calculate final residuals
    total_in_flow = gt.incident_edges_op(graph, "in", "sum", flow)
    in_flow_error = depth.a - total_in_flow.a
    total_out_flow = gt.incident_edges_op(graph, "out", "sum", flow)
    out_flow_error = depth.a - total_out_flow.a
    mean_residual_vertex_flow = (in_flow_error + out_flow_error) / 2
    return flow, mean_residual_vertex_flow, loss_hist
