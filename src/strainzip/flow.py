import logging
import warnings

import graph_tool as gt
import numpy as np

from .logging_util import tqdm_debug


def _calculate_static_terms(graph, depth, length, alpha):
    depth_source = gt.edge_endpoint_property(graph, depth, "source")
    depth_target = gt.edge_endpoint_property(graph, depth, "target")
    length_source = gt.edge_endpoint_property(graph, length, "source")
    length_target = gt.edge_endpoint_property(graph, length, "target")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
        )
        length_frac_source = graph.new_edge_property(
            "float",
            vals=length_source.a ** (alpha)
            / (length_source.a ** (alpha) + length_target.a ** (alpha)),
        )
        length_frac_target = graph.new_edge_property(
            "float", vals=1 - length_frac_source.a
        )
    return (
        depth_source,
        depth_target,
        length_source,
        length_target,
        length_frac_source,
        length_frac_target,
    )


def _preallocated_terms(graph):
    total_outflow = graph.new_vertex_property("float")
    total_inflow = graph.new_vertex_property("float")
    total_outflow_source = graph.new_edge_property("float")
    total_inflow_target = graph.new_edge_property("float")
    return total_outflow, total_inflow, total_outflow_source, total_inflow_target


def _calculate_delta(flow, graph, depth, static_terms, preallocated_terms):
    # These terms are all pre-computed/pre-allocated
    (
        depth_source,
        depth_target,
        length_source,
        length_target,
        length_frac_source,
        length_frac_target,
    ) = static_terms
    (
        total_outflow,
        total_inflow,
        total_outflow_source,
        total_inflow_target,
    ) = preallocated_terms

    gt.incident_edges_op(graph, "out", "sum", flow, total_outflow)
    gt.incident_edges_op(graph, "in", "sum", flow, total_inflow)
    gt.edge_endpoint_property(graph, total_outflow, "source", total_outflow_source)
    gt.edge_endpoint_property(graph, total_inflow, "target", total_inflow_target)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
        )
        out_fraction_source = np.nan_to_num(flow.a / total_outflow_source.a, nan=1)
        in_fraction_target = np.nan_to_num(flow.a / total_inflow_target.a, nan=1)
    error_source = depth_source.a - total_outflow_source.a
    error_target = depth_target.a - total_inflow_target.a

    correction = graph.new_edge_property(
        "float",
        vals=(error_source * out_fraction_source * length_frac_source.a)
        + (error_target * in_fraction_target * length_frac_target.a),
    )

    # TODO: Drop this
    if np.isnan(correction.a).any():
        print("NaN in flow step.")
        print("Start *_calculate_delta* DEBUG:")
        print(np.isnan(flow.a).sum())
        print(np.isnan(depth.a).sum())
        print(np.isnan(correction.a).sum())
        print(np.isnan(error_source).sum())
        print(np.isnan(error_target).sum())
        print(np.isnan(out_fraction_source).sum())
        print(np.isnan(in_fraction_target).sum())
        print(np.isnan(length_frac_source.a).sum())
        print(np.isnan(length_frac_target.a).sum())
        print(graph.vp["filter"].a.sum())
        print((graph.vp["length"].a == 0).sum())
        print(np.isnan(correction.fa).sum())
        print(graph)
        print("End *_calculate_delta* DEBUG:")

    return correction


def estimate_flow(
    graph,
    depth,
    length,
    eps=1e-6,
    maxiter=1000,
    flow_init=None,
    ifnotconverged="warn",
):
    assert ifnotconverged in ["ignore", "warn", "error"]
    if flow_init is not None:
        flow = flow_init
    else:
        flow = graph.new_edge_property("float", val=1)

    preallocated_terms = _preallocated_terms(graph)
    loss_hist = []

    # First: Ignore length
    # NOTE: static_terms constructs the *length_frac_XXX* as 1/n
    # when alpha=0
    # FIXME (2024-05-10): Clean up the double progress bar.
    static_terms = _calculate_static_terms(graph, depth, length, alpha=0)
    pbar1 = tqdm_debug(
        range(maxiter),
        total=maxiter,
        bar_format="{l_bar}{r_bar}",
    )
    for _ in pbar1:
        correction = _calculate_delta(
            flow, graph, depth, static_terms, preallocated_terms
        )
        loss_hist.append(np.sqrt((correction.fa**2).sum()) / depth.fa.sum())
        if np.isnan(loss_hist[-1]):
            raise RuntimeError("NaN during flow estimation.")
        elif loss_hist[-1] == 0:
            break  # This should only happen if d is all 0's.

        flow.a += correction.a
        pbar1.set_postfix({"relative_loss": loss_hist[-1]})
        if loss_hist[-1] < eps:
            break
    else:
        if ifnotconverged == "warn":
            warnings.warn("Reached maxiter. Flow estimates did not converge.")
        elif ifnotconverged == "error":
            raise RuntimeError("Reached maxiter. Flow estimates did not converge.")
        elif ifnotconverged == "ignore":
            pass

    # Second: Weight by length
    # NOTE: static_terms constructs the *length_frac_XXX* as you would expect
    # when alpha=1
    static_terms = _calculate_static_terms(graph, depth, length, alpha=1)
    pbar2 = tqdm_debug(
        range(maxiter),
        total=maxiter,
        bar_format="{l_bar}{r_bar}",
    )
    for _ in pbar2:
        correction = _calculate_delta(
            flow, graph, depth, static_terms, preallocated_terms
        )
        # FIXME: Overflow on square of the correction.
        loss_hist.append(np.sqrt((correction.fa**2).sum()) / depth.fa.sum())
        if np.isnan(loss_hist[-1]):
            raise RuntimeError("NaN during flow estimation.")
        elif loss_hist[-1] == 0:
            break  # This should only happen if d is all 0's.

        flow.a += correction.a
        pbar2.set_postfix({"relative_loss": loss_hist[-1]})
        if loss_hist[-1] < eps:
            break
    else:
        if ifnotconverged == "warn":
            warnings.warn("Reached maxiter. Flow estimates did not converge.")
        elif ifnotconverged == "error":
            raise RuntimeError("Reached maxiter. Flow estimates did not converge.")
        elif ifnotconverged == "ignore":
            pass

    return flow, loss_hist


def estimate_flow_old(
    graph,
    depth,
    length,
    eps=0.001,
    maxiter=1000,
    flow_init=None,
    ifnotconverged="warn",
):
    assert ifnotconverged in ["ignore", "warn", "error"]

    target_vertex_length = gt.edge_endpoint_property(graph, length, "target")
    source_vertex_length = gt.edge_endpoint_property(graph, length, "source")

    # Allocate PropertyMaps
    if flow_init is not None:
        flow = flow_init
    else:
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

    loss_hist = []
    i = 0
    pbar = tqdm_debug(
        range(maxiter),
        total=maxiter,
        mininterval=1.0,
        bar_format="{l_bar}{r_bar}",
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
        if i > 1:
            loss_ratio = (loss_hist[-2] - loss_hist[-1]) / loss_hist[-2]
            pbar.set_postfix({"loss": loss_hist[-1], "improvement": loss_ratio})
            if loss_ratio < eps:
                break

        # NOTE: Some values of (source_vertex_length.a + target_vertex_length.a)
        # are 0 because these two edge_properties include edge indices
        # for non-existent edges.
        # TODO: Consider running gt.reindex_edges to get rid of these.
        mean_flow_error = (
            (source_vertex_alloc_error * source_vertex_length.a)
            + (target_vertex_alloc_error * target_vertex_length.a)
        ) / (source_vertex_length.a + target_vertex_length.a)
        flow.a += mean_flow_error
    else:
        if ifnotconverged == "warn":
            warnings.warn("Reached maxiter. Flow estimates did not converge.")
        elif ifnotconverged == "error":
            raise RuntimeError("Reached maxiter. Flow estimates did not converge.")
        elif ifnotconverged == "ignore":
            pass

    return flow, loss_hist


def calculate_mean_residual_vertex_flow(graph, flow, depth):
    total_in_flow = gt.incident_edges_op(graph, "in", "sum", flow)
    in_flow_error = depth.a - total_in_flow.a
    total_out_flow = gt.incident_edges_op(graph, "out", "sum", flow)
    out_flow_error = depth.a - total_out_flow.a
    mean_residual_vertex_flow = (in_flow_error + out_flow_error) / 2
    return mean_residual_vertex_flow


def estimate_all_flows(
    graph,
    eps=1e-6,
    maxiter=1000,
    flow_init=None,
    ifnotconverged="warn",
):
    depth_list = gt.ungroup_vector_property(
        graph.vp["depth"], pos=range(graph.gp["num_samples"])
    )
    length = graph.vp["length"]

    flow = []
    for depth in depth_list:
        flow.append(
            estimate_flow(
                graph,
                depth,
                length,
                eps=eps,
                maxiter=maxiter,
                ifnotconverged=ifnotconverged,
            )[0]
        )
    flow = gt.group_vector_property(flow)
    return flow


def estimate_depth(graph, flow):
    total_in_flow = graph.degree_property_map("in", weight=flow)
    total_in_degree = graph.degree_property_map("in")
    total_out_flow = graph.degree_property_map("out", weight=flow)
    total_out_degree = graph.degree_property_map("out")

    twoway_flow = total_in_flow.a + total_out_flow.a
    # If in_degree or out_degree == 0, then don't take the mean of both, but rather just the "mean" of one.
    mean_flow = np.where(
        (total_in_degree.a > 0) & (total_out_degree.a > 0),
        twoway_flow / 2,
        twoway_flow / 1,
    )
    return graph.new_vertex_property("float", vals=mean_flow)


def smooth_depth(
    graph,
    depth_init,
    length,
    eps=1e-6,
    maxiter=1000,
    estimate_flow_kwargs=None,
    ifnotconverged="warn",
):
    assert ifnotconverged in ["ignore", "warn", "error"]
    # Set kwargs for estimate flow.
    if estimate_flow_kwargs is None:
        estimate_flow_kwargs = {}

    depth = depth_init.copy()

    loss_hist = []
    pbar = tqdm_debug(
        range(maxiter),
        total=maxiter,
        bar_format="{l_bar}{r_bar}",
    )
    for _ in pbar:
        flow, _ = estimate_flow(graph, depth, length, **estimate_flow_kwargs)
        next_depth = estimate_depth(graph, flow)
        change_in_depth = next_depth.fa - depth.fa
        # FIXME: Overflow on square of the correction.
        loss_hist.append(np.sqrt((change_in_depth**2).sum()) / depth.fa.sum())
        if np.isnan(loss_hist[-1]):
            raise RuntimeError("NaN during depth estimation.")
        elif loss_hist[-1] == 0:
            break  # TODO: Ask if/when this happens.

        depth = next_depth
        pbar.set_postfix
        if loss_hist[-1] < eps:
            break
    else:
        if ifnotconverged == "warn":
            warnings.warn("Reached maxiter. Depth estimates did not converge.")
        elif ifnotconverged == "error":
            raise RuntimeError("Reached maxiter. Depth estimates did not converge.")
        elif ifnotconverged == "ignore":
            pass

    return depth, loss_hist
