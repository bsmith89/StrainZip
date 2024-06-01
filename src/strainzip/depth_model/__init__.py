from typing import Any, Mapping

from ._huber import HuberDepthModel
from ._laplace import LaplaceDepthModel
from ._laplace_pooled import LaplacePooledDepthModel
from ._normal import NormalDepthModel
from ._normal_pooled import NormalPooledDepthModel
from ._offset_log_normal import OffsetLogNormalDepthModel
from ._studentst import StudentsTDepthModel

NAMED_DEPTH_MODELS: Mapping[str, Any] = {
    "OffsetLogNormal": (OffsetLogNormalDepthModel, dict(alpha=1.0)),
    "Normal": (NormalDepthModel, dict()),
    "NormalPooled": (NormalPooledDepthModel, dict()),
    "Laplace": (
        LaplaceDepthModel,
        dict(),
    ),
    "LaplacePooled": (
        LaplacePooledDepthModel,
        dict(),
    ),
    "StudentsT": (
        StudentsTDepthModel,
        dict(df=5),
    ),  # FIXME (2024-05-31): Lots of convergence errors (maybe due to optimizing loc and scale together?).
    "Huber": (HuberDepthModel, dict(delta=1)),
    "Default": (OffsetLogNormalDepthModel, dict(alpha=1.0)),
}
