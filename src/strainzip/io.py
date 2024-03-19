import graph_tool as gt


def parse_bcalm_fasta_entry_header(header, k):
    # for each header, parse out the unitig ID and all the canonical edges.
    # Each unitig has a + and a - version, with <unitig-ID>+/- vertex IDs.
    # Create all of the edges for each of these.
    # ><id> LN:i:<length> KC:i:<abundance> km:f:<abundance> L:<+/->:<other id>:<+/-> [..]
    unitig_id_string, length_string, _, _, *edge_strings_list = header.split()
    unitig_id = unitig_id_string[1:]  # Drop leading >
    length = int(length_string[len("LN:i:") :]) - k + 1  # Overlap of length k-1.

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


def iter_bcalm_fasta_entries(lines_iter):
    header = None
    sequence = None
    for line in lines_iter:
        if line.startswith(">"):
            if header is not None:
                yield (header, sequence)
            header = line
        else:
            sequence = line
    yield (header, sequence)


def parse_bcalm_fasta(lines_iter, k):
    sequences = {}
    lengths = {}
    all_edge_list = []
    for header, sequence in iter_bcalm_fasta_entries(lines_iter):
        unitig_id, length, edge_list = parse_bcalm_fasta_entry_header(header, k=k)
        sequences[unitig_id] = sequence
        lengths[unitig_id] = length
        all_edge_list.extend(edge_list)
    return list(set(all_edge_list)), lengths, sequences


def load_graph_and_sequences_from_bcalm(file_handle, k):
    edge_list, lengths, sequences = parse_bcalm_fasta(file_handle, k=k)

    graph = gt.Graph(set(edge_list), hashed=True, directed=True)
    graph.vp["filter"] = graph.new_vertex_property("bool", val=1)
    graph.vp["length"] = graph.new_vertex_property("int")
    for i, _hash in enumerate(graph.vp["ids"]):
        graph.vp["length"].a[i] = lengths[_hash[:-1]]
    graph.vp["sequence"] = graph.vp["ids"]
    del graph.vp["ids"]
    return graph, sequences


def load_kmer_depths():
    pass
