import graph_tool as gt
import numpy as np
from tqdm import tqdm

from .sequence import iter_kmers, reverse_complement


def bcalm_header_tokenizer(header):
    # for each header, parse out the unitig ID and all the canonical edges.
    # ><id> LN:i:<length> KC:i:<abundance> km:f:<abundance> L:<+/->:<other id>:<+/-> [..]
    unitig_id_string, length_string, _, _, *edge_strings_list = header.split()
    return unitig_id_string, length_string, edge_strings_list


def ggcat_header_tokenizer(header):
    unitig_id_string, length_string, *edge_strings_list = header.split()
    return unitig_id_string, length_string, edge_strings_list


def generic_header_tokenizer(header):
    unitig_id_string, *other_fields = header.split()
    return unitig_id_string, *other_fields


def parse_linked_fasta_entry_header(
    unitig_id_string, length_string, edge_strings_list, k, header_tokenizer
):
    # Each unitig has a + and a - version, with <unitig-ID>+/- vertex IDs.
    # Create all of the edges for each of these.

    length = int(length_string[len("LN:i:") :]) - k + 1  # Overlap of length k-1.
    unitig_id = unitig_id_string[1:]

    edge_list = []
    for edge_string in edge_strings_list:
        unitigA = unitig_id
        _, orientationA, unitigB, orientationB = edge_string.split(":", maxsplit=3)

        if (orientationA, orientationB) == ("+", "+"):
            edge_list.extend(
                [
                    (unitigA + "+", unitigB + "+"),
                    # NOTE: These reverse edges are only necessary if the graph file format is missing them.
                    # (unitigB + '-', unitigA + '-'),
                ]
            )
        elif (orientationA, orientationB) == ("+", "-"):
            edge_list.extend(
                [
                    (unitigA + "+", unitigB + "-"),
                    # (unitigB + '-', unitigA + '+'),
                ]
            )
        elif (orientationA, orientationB) == ("-", "+"):
            edge_list.extend(
                [
                    (unitigA + "-", unitigB + "+"),
                    # (unitigB + '+', unitigA + '-'),
                ]
            )
        elif (orientationA, orientationB) == ("-", "-"):
            edge_list.extend(
                [
                    (unitigA + "-", unitigB + "-"),
                    # (unitigB + '+', unitigA + '+'),
                ]
            )
        else:
            assert False
    return unitig_id, length, edge_list


def iter_linked_fasta_entries(lines_iter):
    header = None
    sequence = None
    for line in lines_iter:
        if line.startswith(">"):
            if header is not None:
                yield (header, sequence)
            header = line
        else:
            sequence = line.strip()
    yield (header, sequence)


def parse_linked_fasta(lines_iter, k, header_tokenizer, verbose=False):
    sequences = {}
    lengths = {}
    all_edge_list = []
    orphan_unitigs = []
    for header, sequence in tqdm(
        iter_linked_fasta_entries(lines_iter), disable=(not verbose)
    ):
        unitig_id_string, length_string, edge_strings_list = header_tokenizer(header)
        unitig_id, length, edge_list = parse_linked_fasta_entry_header(
            unitig_id_string, length_string, edge_strings_list, k, header_tokenizer
        )
        sequences[unitig_id] = sequence
        lengths[unitig_id] = length
        all_edge_list.extend(edge_list)
        if not edge_list:
            orphan_unitigs.extend([unitig_id + "+", unitig_id + "-"])
    return list(set(all_edge_list)), orphan_unitigs, lengths, sequences


def load_graph_and_sequences_from_linked_fasta(
    file_handle, k, header_tokenizer, verbose=False
):
    edge_list, orphan_unitigs, lengths, sequences = parse_linked_fasta(
        file_handle,
        k,
        header_tokenizer,
        verbose=verbose,
    )

    graph = gt.Graph(set(edge_list), hashed=True, directed=True)
    num_non_orphan_unitigs = len(graph)
    graph.add_vertex(n=len(orphan_unitigs))
    for (i, unitig_id) in tqdm(
        enumerate(orphan_unitigs, start=num_non_orphan_unitigs), disable=(not verbose)
    ):
        graph.vp["ids"][i] = unitig_id

    graph.vp["filter"] = graph.new_vertex_property("bool", val=1)
    graph.vp["length"] = graph.new_vertex_property("int")
    for i, _hash in tqdm(enumerate(graph.vp["ids"]), disable=(not verbose)):
        graph.vp["length"].a[i] = lengths[_hash[:-1]]
    graph.vp["sequence"] = graph.vp["ids"]
    del graph.vp["ids"]

    graph.set_vertex_filter(graph.vp["filter"])
    return graph, sequences


def load_mean_sequence_depth(con, sequence, k, n):
    query = "SELECT * FROM count_ WHERE kmer IN (?, ?)"
    num_kmers = len(sequence) - k + 1
    total_tally = np.zeros(n)
    for kmer in iter_kmers(sequence, k=k, circularize=False):
        kmer_rc = reverse_complement(kmer)
        this_kmer_results = con.execute(query, (kmer, kmer_rc)).fetchall()
        if len(this_kmer_results) == 1:
            # The kmer was observed. Add the counts across n samples to the accumulator.
            this_kmer_results = this_kmer_results[0][1:]
            total_tally += np.array(this_kmer_results)
        elif len(this_kmer_results) == 0:
            # The kmer was not observed
            continue
        else:
            raise RuntimeError(
                "Only one of a kmer and it's reverse-complement should be in the database."
            )
    return total_tally / num_kmers


def dump_graph(graph, path, purge=False):
    graph = gt.Graph(graph, prune=purge)
    graph.save(path, fmt="gt")


def load_graph(path):
    graph = gt.load_graph(path, fmt="gt")
    graph.set_vertex_filter(graph.vp["filter"])
    # TODO (2024-05-15): Add filter as a default edge property.
    # graph.set_edge_filter(graph.vp["filter"])
    return graph
