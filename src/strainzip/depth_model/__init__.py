from typing import Any, Mapping

from ._laplace import LaplaceDepthModel
from ._normal import NormalDepthModel
from ._offset_log_normal import OffsetLogNormalDepthModel
from ._studentst import StudentsTDepthModel

NAMED_DEPTH_MODELS: Mapping[str, Any] = {
    "OffsetLogNormal": (OffsetLogNormalDepthModel, dict(alpha=1.0)),
    "Normal": (NormalDepthModel, dict()),
    "Laplace": (LaplaceDepthModel, dict()),
    "StudentsT": (StudentsTDepthModel, dict(df=5)),
    "Default": (OffsetLogNormalDepthModel, dict(alpha=1.0)),
}
