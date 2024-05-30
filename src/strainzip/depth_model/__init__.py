from typing import Any, Mapping

from ._offset_log_normal import OffsetLogNormalDepthModel
from ._softplus_normal import SoftplusNormalDepthModel
from ._laplace import LaplaceDepthModel
from ._studentst import StudentsTDepthModel

NAMED_DEPTH_MODELS: Mapping[str, Any] = {
    "OffsetLogNormal": (OffsetLogNormalDepthModel, dict(alpha=1.0)),
    "SoftplusNormal": (SoftplusNormalDepthModel, dict()),
    "Laplace": (LaplaceDepthModel, dict()),
    "StudentsT": (StudentsTDepthModel, dict(df=5)),
    "Default": (OffsetLogNormalDepthModel, dict(alpha=1.0)),
}
