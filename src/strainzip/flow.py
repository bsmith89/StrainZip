import warnings

import graph_tool as gt
import numpy as np
from tqdm import tqdm


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
    # NOTE (2024-05-28): This error is due to edges with zero flow going to nodes with zero total
    # flow. We redefine *_fraction_* to be 1, but this doesn't really matter;
    # That 1 is later multiplied by error_*, and should have no effect on the final
    # "correction" value... Yeah...?
    # NOTE (2024-05-28): If this code is run multithreaded (not but not with multiprocessing
    # or single-threaded) the catch_warnings context manager can do weird things.
    # While I could replace this with manual setting/resetting filters, I'm almost
    # never using threading here. (Although I am often using multiprocessing.)
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
        raise RuntimeError()

    return correction


def estimate_flow(
    graph,
    depth,
    length,
    eps=1e-6,
    maxiter=1000,
    flow_init=None,
    ifnotconverged="warn",
    verbose=False,
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
    pbar1 = tqdm(
        range(maxiter),
        total=maxiter,
        bar_format="{l_bar}{r_bar}",
        disable=(not verbose),
    )
    for _ in pbar1:
        correction = _calculate_delta(
            flow, graph, depth, static_terms, preallocated_terms
        )
        # FIXME: Overflow on square of the correction when using multiprocessing.
        # FIXME (2024-05-28): I may have solved this with clipping.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=RuntimeWarning,
            )
            loss_hist.append(np.sqrt((correction.fa**2).sum()) / depth.fa.sum())
        if np.isnan(loss_hist[-1]):
            raise RuntimeError("NaN during flow estimation.")
        elif loss_hist[-1] == 0:
            break  # This should only happen if d is all 0's.

        # Update flow
        flow.a += correction.a
        # NOTE (2024-05-28):
        # Very small negative values can show up (numerical precision issues?)
        # which then baloon into large negative values.
        # Clipping seems to solve this.
        flow.a = np.clip(flow.a, 0, None)

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
    pbar2 = tqdm(
        range(maxiter),
        total=maxiter,
        bar_format="{l_bar}{r_bar}",
        disable=(not verbose),
    )
    for _ in pbar2:
        correction = _calculate_delta(
            flow, graph, depth, static_terms, preallocated_terms
        )
        # FIXME: Overflow on square of the correction when using multiprocessing.
        # FIXME (2024-05-28): I may have solved this with clipping.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=RuntimeWarning,
            )
            loss_hist.append(np.sqrt((correction.fa**2).sum()) / depth.fa.sum())
        if np.isnan(loss_hist[-1]):
            raise RuntimeError("NaN during flow estimation.")
        elif loss_hist[-1] == 0:
            break  # This should only happen if d is all 0's.

        # Update flow
        flow.a += correction.a
        # NOTE (2024-05-28):
        # Very small negative values can show up (numerical precision issues?)
        # which then baloon into large negative values.
        # Clipping seems to solve this.
        flow.a = np.clip(flow.a, 0, None)

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


def estimate_depth(graph, flow, pseudoflow):
    total_in_flow = graph.degree_property_map("in", weight=flow)
    total_in_degree = graph.degree_property_map("in")
    total_out_flow = graph.degree_property_map("out", weight=flow)
    total_out_degree = graph.degree_property_map("out")

    # If in_degree or out_degree == 0, then replace the inflow with the pseudoflow.
    in_flow = np.where(total_in_degree.a > 0, total_in_flow.a, pseudoflow.a)
    out_flow = np.where(total_out_degree.a > 0, total_out_flow.a, pseudoflow.a)
    mean_flow = (in_flow + out_flow) / 2
    # TODO (2024-05-22): Consider whether this results in a problematic bias towards higher
    # depth for tips and therefore more weight going into unzipped paths.
    return graph.new_vertex_property("float", vals=mean_flow)


def calculate_vertex_pseudoflow(graph, depth, length):
    return graph.new_vertex_property(
        "float", vals=(depth.a * length.a) / (length.a + 1)
    )


def smooth_depth(
    graph,
    depth_init,
    length,
    eps=1e-6,
    maxiter=1000,
    estimate_flow_kwargs=None,
    ifnotconverged="warn",
    verbose=False,
):
    assert ifnotconverged in ["ignore", "warn", "error"]
    # Set kwargs for estimate flow.
    if estimate_flow_kwargs is None:
        estimate_flow_kwargs = {}

    # NOTE: The pseudoflow is a value for every vertex based on an imaginary
    # edge connecting it to a length-1 unitig with depth 0.
    # In order to estimate the depth of vertices with 0 in-degree or 0 out-degree,
    # we'll replace the in/out-flow with this value.
    pseudoflow = calculate_vertex_pseudoflow(graph, depth_init, length)

    depth = depth_init.copy()

    loss_hist = []
    pbar = tqdm(
        range(maxiter),
        total=maxiter,
        bar_format="{l_bar}{r_bar}",
        disable=(not verbose),
    )
    for _ in pbar:
        flow, _ = estimate_flow(graph, depth, length, **estimate_flow_kwargs)
        next_depth = estimate_depth(graph, flow, pseudoflow)
        correction = next_depth.fa - depth.fa
        # FIXME: Overflow on square of the correction when using multiprocessing.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=RuntimeWarning,
            )
            loss_hist.append(np.sqrt((correction**2).sum()) / depth.fa.sum())
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
