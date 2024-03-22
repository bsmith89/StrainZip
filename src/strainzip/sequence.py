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