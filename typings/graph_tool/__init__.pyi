from typing import (
    Any,
    Generic,
    List,
    MutableMapping,
    NewType,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from numpy.typing import NDArray

from . import clustering, draw, generation, topology

ScalarT = TypeVar("ScalarT", bound=Any)

class Vertex: ...

class Graph:
    def __init__(self, graph=None, directed=True, hashed=False, prune=False): ...
    vertex_properties: MutableMapping[str, VertexPropertyMap]
    vp: MutableMapping[str, VertexPropertyMap]
    ep: MutableMapping[str, EdgePropertyMap]
    gp: MutableMapping[str, Any]
    def purge_vertices(self) -> None: ...
    def purge_edges(self) -> None: ...
    vertex_index: VertexPropertyMap
    def add_vertex(self, n=1): ...
    def add_edge_list(self, edge_list: List[Tuple[int, int]]) -> None: ...
    def new_vertex_property(
        self, return_type, vals=None, val=None
    ) -> VertexPropertyMap: ...
    def new_edge_property(
        self, return_type, vals=None, val=None
    ) -> EdgePropertyMap: ...
    def new_graph_property(self, return_type, val=None) -> GraphPropertyMap: ...
    def __len__(self) -> int: ...
    def get_in_neighbors(self, v, vprops=Sequence[VertexPropertyMap]) -> NDArray: ...
    def get_out_neighbors(self, v, vprops=Sequence[VertexPropertyMap]) -> NDArray: ...
    def num_vertices(self, ignore_filter=False) -> int: ...
    def num_edges(self) -> int: ...
    def set_vertex_filter(self, VertexPropertyMap) -> None: ...
    def set_edge_filter(self, EdgePropertyMap) -> None: ...
    def get_in_degrees(self, vs) -> NDArray: ...
    def get_out_degrees(self, vs) -> NDArray: ...
    def get_vertices(self, vprops=[]) -> NDArray: ...
    def get_edges(self, eprops=[]) -> NDArray: ...
    def degree_property_map(self, deg, weight=None) -> VertexPropertyMap: ...
    def save(self, file_name, fmt="auto") -> None: ...
    def own_property(self, prop: PropertyMap) -> PropertyMap: ...
    def get_vertex_filter(self) -> Tuple[VertexPropertyMap, bool]: ...
    def shrink_to_fit(self) -> None: ...

class GraphView(Graph):
    def __init__(
        self,
        g,
        vfilt=None,
        efilt=None,
        directed=None,
        reversed=False,
        skip_properties=False,
        skip_vfilt=False,
        skip_efilt=False,
    ): ...

class PropertyMap(Generic[ScalarT]):
    def copy(self) -> PropertyMap: ...
    def value_type(self) -> str: ...
    @property
    def a(self) -> NDArray: ...
    @property
    def fa(self) -> NDArray: ...
    @property
    def ma(self) -> NDArray: ...
    def __iter__(self): ...
    def get_2d_array(self, pos: Sequence[int]) -> NDArray: ...
    def set_2d_array(self, a, pos=None) -> None: ...

class VertexPropertyMap(PropertyMap):
    def __getitem__(self, i: int) -> Any: ...
    def __setitem__(self, i: int, val: Any) -> None: ...

class EdgePropertyMap(PropertyMap):
    def __getitem__(self, i: Tuple[int, int]) -> Any: ...
    def __setitem__(self, i: Tuple[int, int], val: Any) -> None: ...

class GraphPropertyMap(PropertyMap): ...

PropMapT = TypeVar("PropMapT", bound=PropertyMap)

def group_vector_property(
    props: Sequence[PropMapT],
    value_type: Optional[str] = None,
    pos: Optional[Sequence[int]] = None,
    vprop: Optional[VertexPropertyMap] = None,
) -> PropMapT: ...
def ungroup_vector_property(
    vprop: PropMapT, pos=None, props=None
) -> List[PropMapT]: ...
def seed_rng(seed: int) -> None: ...
def edge_endpoint_property(g, prop, endpoint, eprop=None) -> EdgePropertyMap: ...
def incident_edges_op(g, direction, op, eprop, vprop=None) -> VertexPropertyMap: ...
def load_graph(
    file_name, fmt="auto", ignore_vp=None, ignore_ep=None, ignore_gp=None
) -> Graph: ...
def map_property_values(
    src_prop: PropertyMap, tgt_prop: PropertyMap, map_func: Any
) -> None: ...
def openmp_set_num_threads(n: int) -> None: ...
