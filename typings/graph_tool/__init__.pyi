from typing import Any, Generic, List, NewType, Self, TypeVar

from numpy.typing import NDArray

Vertex = NewType("Vertex", object)
ScalarT = TypeVar("ScalarT", bound=Any)

class Graph:
    vertices: List[Vertex]

class VertexPropertyMap(Generic[ScalarT]):
    ...

    def __getitem__(self, i: int) -> ScalarT: ...
    def __setitem__(self, i: int, val: ScalarT) -> None: ...
    @property
    def a(self) -> NDArray: ...
