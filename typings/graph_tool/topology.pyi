from typing import Iterator, List, Tuple

from numpy.typing import NDArray

from . import VertexPropertyMap

def label_components(
    g, vprop=None, directed=None
) -> Tuple[VertexPropertyMap, NDArray]: ...
def topological_sort(g) -> NDArray: ...
def is_DAG(g) -> bool: ...
def all_paths(g, source, target, cutoff=None, edges=False) -> Iterator[List[int]]: ...
def shortest_distance(
    g,
    source=None,
    target=None,
    weights=None,
    negative_weights=False,
    max_dist=None,
    directed=None,
    dense=False,
    dist_map=None,
    pred_map=False,
    return_reached=False,
    dag=False,
): ...
def all_circuits(g, unique=False) -> Iterator[List[int]]: ...
