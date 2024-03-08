from typing import List, NewType, Self

Vertex = NewType("Vertex", object)

class Graph:
    vertices: List[Vertex]
