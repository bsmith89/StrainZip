from collections import Counter, defaultdict
from itertools import chain, product

import graph_tool as gt
import graph_tool.draw
import numpy as np

COMPLEMENTARY_BASE = {"A": "T", "C": "G", "G": "C", "T": "A"}


def random_sequence(length):
    return "".join(np.random.choice(["A", "C", "G", "T"], size=length, replace=True))


def base_pair(b):
    return COMPLEMENTARY_BASE[b]


def reverse_complement(s):
    return "".join(map(base_pair, reversed(s)))


def iter_kmers(s, k, circularize=False):
    if circularize:
        # Append a k-1-prefix of sequence s to the end of s.
        s = s + s[: k - 1]

    for i in range(0, len(s) - k + 1):
        yield s[i : i + k]


def iter_dbg_edges(kmer_iter):
    sfx = defaultdict(list)
    pfx = defaultdict(list)
    for kmer in kmer_iter:
        sfx[kmer[1:]].append(kmer)
        pfx[kmer[:-1]].append(kmer)
    shared = set(sfx) & set(pfx)
    for km1 in shared:
        for left, right in product(sfx[km1], pfx[km1]):
            yield (left, right)


def annotated_dbg(sequence, k, circularize=False, include_rc=False, position=False):
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
    if position:
        graph.vp["xyposition"] = gt.draw.sfdp_layout(graph)
    graph.vp["filter"] = graph.new_vertex_property("bool", val=True)
    del graph.vp["ids"]
    return graph
