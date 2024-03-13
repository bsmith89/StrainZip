from typing import Tuple

from numpy.typing import NDArray

from . import VertexPropertyMap

def label_components(
    g, vprop=None, directed=None
) -> Tuple[VertexPropertyMap, NDArray]: ...
def topological_sort(g) -> NDArray: ...
