from typing import Any, List, Optional

from . import Graph, VertexPropertyMap

def motifs(
    g: Graph,
    k: int,
    p: float = 1.0,
    motif_list: Optional[List[Graph]] = None,
    return_maps: bool = False,
) -> Any: ...
