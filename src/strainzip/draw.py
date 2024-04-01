import graph_tool as gt
import graph_tool.draw
import numpy as np


def update_xypositions(graph, layout=gt.draw.sfdp_layout, init_pos=None, **kwargs):
    if ("xyposition" in graph.vp) and (init_pos is None):
        init_pos = graph.vp["xyposition"]

    graph.vp["xyposition"] = layout(graph, pos=init_pos, **kwargs)


def draw_graph(graph, **kwargs):
    draw_kwargs = dict(
        pos=None, ink_scale=0.35, vertex_shape="square", vertex_text=graph.vertex_index
    )

    if "xyposition" in graph.vp:
        draw_kwargs["pos"] = graph.vp["xyposition"]
    if "depth" in graph.vp:
        if graph.vp["depth"].value_type() == "double":
            draw_kwargs["vertex_fill_color"] = graph.vp["depth"]
        elif "length" in graph.vp:
            draw_kwargs["vertex_fill_color"] = graph.vp["length"]

    return gt.draw.graph_draw(
        graph,
        **(draw_kwargs | kwargs),
    )
