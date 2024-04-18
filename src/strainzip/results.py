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


def extract_vertex_data(graph, segment_to_sequence):
    vertex_data = (
        pd.DataFrame(
            dict(
                vertex=graph.get_vertices(),
                length=graph.vp["length"],
                total_depth=graph.vp["depth"]
                .get_2d_array(range(graph.gp["num_samples"]))
                .sum(0),
                segments=[ss.split(",") for ss in graph.vp["sequence"]],
                in_neighbors=[
                    frozenset(graph.get_in_neighbors(v)) for v in graph.get_vertices()
                ],
                out_neighbors=[
                    frozenset(graph.get_out_neighbors(v)) for v in graph.get_vertices()
                ],
            )
        )
        .assign(
            segments=lambda x: x.segments.apply(tuple),
            num_segments=lambda x: x.segments.apply(len),
            assembly=lambda x: x.segments.apply(
                lambda y: assemble_overlapping_unitigs(
                    y, segment_to_sequence, k=graph.gp["kmer_length"]
                )
            ),
        )
        .set_index("vertex")
    )
    assert (
        vertex_data.assembly.apply(len)
        == vertex_data.length + graph.gp["kmer_length"] - 1
    ).all()

    return vertex_data


def deduplicate_vertex_data(vertex_data):
    deduplicated_data = (
        vertex_data.sort_values("total_depth", ascending=False)
        .reset_index()
        .groupby("assembly")
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
    twins = []
    for vertex, d1 in vertex_data.iterrows():
        reversed_twin_segments = []
        for segment in d1.segments:
            unitig, strand = segment[:-1], segment[-1]
            opposite_strand = {"+": "-", "-": "+"}[strand]
            reversed_twin_segments.append(f"{unitig}{opposite_strand}")
        twin_segments = tuple(reversed(reversed_twin_segments))
        twin_vertices = idxwhere(vertex_data.segments == twin_segments)
        assert len(twin_vertices) == 1
        twins.append((vertex, twin_vertices[0]))

    return twins


def validate_twins(twins, vertex_data):
    # Check that all twins have the same info:
    for i, j in twins:
        a = vertex_data.loc[i]
        b = vertex_data.loc[j]
        assert (len(a.in_neighbors), len(a.out_neighbors)) == (
            len(b.out_neighbors),
            len(b.in_neighbors),
        )
        assert a.length == b.length
        assert len(a.segments) == len(b.segments)
        assert a.total_depth == b.total_depth
