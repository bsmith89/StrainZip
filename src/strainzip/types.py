from typing import NewType

__all__ = ["VertexID", "ChildID", "ParentID"]

VertexID = NewType("VertexID", int)
ChildID = NewType("ChildID", VertexID)
ParentID = NewType("ParentID", VertexID)
