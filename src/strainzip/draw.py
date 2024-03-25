import graph_tool as gt
import graph_tool.draw
import numpy as np


def update_xypositions(graph, layout=gt.draw.sfdp_layout, **kwargs):
    if "xyposition" in graph.vp:
        init_pos = graph.vp["xyposition"]
    else:
        init_pos = None

    graph.vp["xyposition"] = layout(graph, pos=init_pos, **kwargs)


def draw_graph(graph, **kwargs):
    draw_kwargs = dict(
        pos=None, ink_scale=0.35, vertex_shape="square", vertex_text=graph.vertex_index
    )

    if "xyposition" in graph.vp:
        draw_kwargs["pos"] = graph.vp["xyposition"]
    if "length" in graph.vp:
        vertex_aspect = graph.vp["length"].copy()
        vertex_aspect.a[:] = np.sqrt(vertex_aspect.a)
        draw_kwargs["vertex_aspect"] = vertex_aspect
    if "depth" in graph.vp:
        if graph.vp["depth"].value_type() == "double":
            draw_kwargs["vertex_fill_color"] = graph.vp["depth"]

    return gt.draw.graph_draw(
        graph,
        **(draw_kwargs | kwargs),
    )
