import graph_tool as gt
import graph_tool.draw
import numpy as np


def draw_graph(graph, filter_vertices=True, **kwargs):
    # FIXME: When the graph is filtered, graph.vp inputs for vertex properties fail (sometimes?).
    if filter_vertices:
        graph = gt.GraphView(graph, vfilt=graph.vp["filter"])

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

    gt.draw.graph_draw(
        graph,
        **(draw_kwargs | kwargs),
    )
