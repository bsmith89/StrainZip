from collections import Counter
from itertools import chain

import graph_tool as gt

from .sequence import iter_dbg_edges, iter_kmers, reverse_complement


def annotate_graph_from_mapping(graph, **kwargs):
    # TODO: Generalized function that can work for both io.load_graph_and_sequences_from_bcalm
    # and annotated_dbg to add lengths, filter, depth, sequence, etc. as vertex
    # properties.
    pass


def annotated_dbg(sequence, k, circularize=False, include_rc=False):
    kmers = iter_kmers(sequence, k=k, circularize=circularize)
    if include_rc:
        kmers = chain(
            kmers,
            iter_kmers(reverse_complement(sequence), k=k, circularize=circularize),
        )
    kmer_counts = Counter(kmers)

    graph = gt.Graph(
        iter_dbg_edges(kmer_counts.keys()),
        directed=True,
        hashed=True,
    )

    graph.vp["length"] = graph.new_vertex_property("int", val=1)
    graph.vp["sequence"] = graph.vp["ids"]  # TODO: .copy()?
    graph.vp["depth"] = graph.new_vertex_property(
        "float", vals=[kmer_counts[k] for k in graph.vp["ids"]]
    )
    graph.vp["filter"] = graph.new_vertex_property("bool", val=True)
    del graph.vp["ids"]  # Drop no-longer-necessary vertex property.
    graph.set_vertex_filter(graph.vp["filter"])
    return graph


def annotate_xypositions(graph, layout=gt.draw.sfdp_layout):
    graph.vp["xyposition"] = layout(graph)
