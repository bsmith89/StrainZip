from typing import Any, Mapping

from ._log_offset_normal import OffsetLogNormalDepthModel

NAMED_DEPTH_MODELS: Mapping[str, Any] = {
    "OffsetLogNormal": (OffsetLogNormalDepthModel, dict(alpha=1.0)),
    "Default": (OffsetLogNormalDepthModel, dict(alpha=1.0)),
}
