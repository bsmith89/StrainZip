from typing import Any, Mapping

from ._offset_log_normal import OffsetLogNormalDepthModel
from ._softplus_normal import SoftplusNormalDepthModel

NAMED_DEPTH_MODELS: Mapping[str, Any] = {
    "OffsetLogNormal": (OffsetLogNormalDepthModel, dict(alpha=1.0)),
    "SoftplusNormal": (SoftplusNormalDepthModel, dict()),
    "Default": (OffsetLogNormalDepthModel, dict(alpha=1.0)),
}
