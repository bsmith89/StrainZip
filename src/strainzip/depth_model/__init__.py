from typing import Any, Mapping

from ._base import DepthModelResult
from ._huber import HuberDepthModel
from ._laplace import LaplaceDepthModel
from ._laplace_pooled import LaplacePooledDepthModel
from ._normal import NormalDepthModel
from ._normal_pooled import NormalPooledDepthModel
from ._normal_scaled2 import NormalScaled2DepthModel
from ._normal_scaled3 import NormalScaled3DepthModel
from ._offset_log_normal import OffsetLogNormalDepthModel
from ._studentst import StudentsTDepthModel

VALIDATED_DEPTH_MODELS: Mapping[str, Any] = {
    "OffsetLogNormal": (OffsetLogNormalDepthModel, dict(alpha=1.0, maxiter=10000)),
    "Default": (OffsetLogNormalDepthModel, dict(alpha=1.0, maxiter=10000)),
}

EXPERIMENTAL_DEPTH_MODELS: Mapping[str, Any] = {
    "Normal": (NormalDepthModel, dict(maxiter=10000, tol=1e-4)),
    "NormalScaled2": (
        NormalScaled2DepthModel,
        dict(alpha=1.0, maxiter=1000, tol=1e-3),
    ),
    "NormalScaled3": (
        NormalScaled3DepthModel,
        dict(alpha=1 / 2, maxiter=1000, tol=1e-3),
    ),
    "NormalPooled": (NormalPooledDepthModel, dict(maxiter=10000)),
    "Laplace": (
        LaplaceDepthModel,
        dict(maxiter=10000),
    ),
    "LaplacePooled": (
        LaplacePooledDepthModel,
        dict(maxiter=10000),
    ),
    "StudentsT": (
        StudentsTDepthModel,
        dict(df=5, maxiter=10000),
    ),  # FIXME (2024-05-31): Lots of convergence errors (maybe due to optimizing loc and scale together?).
    "Huber": (HuberDepthModel, dict(delta=1, maxiter=10000)),
}

NAMED_DEPTH_MODELS = EXPERIMENTAL_DEPTH_MODELS | VALIDATED_DEPTH_MODELS
