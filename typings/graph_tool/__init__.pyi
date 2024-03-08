from typing import List, NewType, Self, TypeAlias

Vertex = NewType("Vertex", object)

class Graph:
    vertices: List[Vertex]
