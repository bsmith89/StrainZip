from itertools import chain

import pandas as pd

from .pandas_util import idxwhere
from .sequence import reverse_complement


def assemble_overlapping_unitigs(segment_list, unitig_to_sequence, k):
    accum = ""
    for segment in segment_list:
        unitig, strand = segment[:-1], segment[-1:]
        forward_segment = unitig_to_sequence[unitig]
        if strand == "-":
            forward_segment = reverse_complement(forward_segment)
        if accum:
            assert accum[-(k - 1) :] == forward_segment[: (k - 1)]
        accum = accum[: -(k - 1)] + forward_segment
    return accum


def extract_vertex_data(graph):
    vertex_data = dict(
        vertex=graph.get_vertices(),
        in_neighbors=[
            frozenset(graph.get_in_neighbors(v)) for v in graph.get_vertices()
        ],
        out_neighbors=[
            frozenset(graph.get_out_neighbors(v)) for v in graph.get_vertices()
        ],
    )
    if "length" in graph.vp:
        vertex_data["length"] = graph.vp["length"]
    if "depth" in graph.vp:
        vertex_data["total_depth"] = (
            graph.vp["depth"].get_2d_array(range(graph.gp["num_samples"])).sum(0)
        )
    if "sequence" in graph.vp:
        vertex_data["segments"] = [ss.split(",") for ss in graph.vp["sequence"]]
    vertex_data = pd.DataFrame(vertex_data).set_index("vertex")

    if "sequence" in graph.vp:
        vertex_data = vertex_data.assign(
            segments=lambda x: x.segments.apply(tuple),
            num_segments=lambda x: x.segments.apply(len),
        )
    vertex_data = vertex_data.assign(
        num_in_neighbors=lambda x: x.in_neighbors.apply(len),
        num_out_neighbors=lambda x: x.out_neighbors.apply(len),
    )
    return vertex_data


def assemble_segments(graph, unitig_sequences):
    vertex_data = (
        pd.DataFrame(
            dict(
                vertex=graph.get_vertices(),
                length=graph.vp["length"],
                segments=[ss.split(",") for ss in graph.vp["sequence"]],
            )
        )
        .assign(
            segments=lambda x: x.segments.apply(tuple),
            assembly=lambda x: x.segments.apply(
                lambda y: assemble_overlapping_unitigs(
                    y, unitig_sequences, k=graph.gp["kmer_length"]
                )
            ),
        )
        .set_index("vertex")
    )
    assert (
        vertex_data.assembly.apply(len)
        == vertex_data.length + graph.gp["kmer_length"] - 1
    ).all()

    return vertex_data.assembly


def deduplicate_vertex_data(vertex_data):
    deduplicated_data = (
        vertex_data.sort_values("total_depth", ascending=False)
        .reset_index()
        .groupby("segments")
        .apply(
            lambda x: pd.Series(
                dict(
                    vertex=x.vertex.iloc[0],
                    length=x.length.iloc[0],
                    total_depth=x.total_depth.sum(),
                    segments=x.segments.iloc[0],
                    vertices=tuple(set(x.vertex)),
                    in_neighbors=tuple(set(chain.from_iterable(x.in_neighbors))),
                    out_neighbors=tuple(set(chain.from_iterable(x.out_neighbors))),
                    num_segments=x.num_segments.iloc[0],
                )
            )
        )
        .reset_index()
        .set_index("vertex")
    )
    return deduplicated_data


def find_twins(vertex_data):
    # Find all twins on the initial graph
    twins = {}
    for vertex, d1 in vertex_data.iterrows():
        reversed_twin_segments = []
        for segment in d1.segments:
            unitig, strand = segment[:-1], segment[-1]
            opposite_strand = {"+": "-", "-": "+"}[strand]
            reversed_twin_segments.append(f"{unitig}{opposite_strand}")
        twin_segments = tuple(reversed(reversed_twin_segments))
        twin_vertices = idxwhere(vertex_data.segments == twin_segments)
        twins[vertex] = list(twin_vertices)
    return twins


def validate_twins(twins, vertex_data):
    # Check that all twins have the same info:
    vertex_data = vertex_data.drop(
        columns=["segments", "in_neighbors", "out_neighbors"]
    )
    # Flatten twins
    twins = [(k, v) for k in twins for v in twins[k]]
    first, second = zip(*twins)
    first_vdata = vertex_data.loc[list(first)]
    second_vdata = (
        vertex_data.loc[list(second)]
        .rename(
            columns={
                "num_in_neighbors": "num_out_neighbors",
                "num_out_neighbors": "num_in_neighbors",
            }
        )
        .rename(dict(twins))
    )
    first_vdata, second_vdata = first_vdata.align(second_vdata)
    comparison = first_vdata - second_vdata
    return (comparison == 0).all().all(), comparison


def iter_vertices_with_segment(graph, segment):
    for v in graph.get_vertices():
        segment_tuple = graph.vp["sequence"][v].split(",")
        if segment in segment_tuple:
            yield v
