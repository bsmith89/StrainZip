from typing import Any, Generic, List, NewType, Sequence, Tuple, TypeVar

from numpy.typing import NDArray

Vertex = NewType("Vertex", object)
ScalarT = TypeVar("ScalarT", bound=Any)

class Graph:
    vertices: List[Vertex]
    def add_vertex(self, n=1): ...
    def add_edge_list(self, edge_list: List[Tuple[int, int]]) -> None: ...
    def new_vertex_property(
        self, return_type, vals=None, val=None
    ) -> VertexPropertyMap: ...

class VertexPropertyMap(Generic[ScalarT]):
    ...

    def __getitem__(self, i: int) -> ScalarT: ...
    def __setitem__(self, i: int, val: ScalarT) -> None: ...
    @property
    def a(self) -> NDArray: ...
    def __iter__(self): ...
    def get_2d_array(self, pos: Sequence[int]) -> NDArray: ...
