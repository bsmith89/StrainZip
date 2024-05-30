from ._log_offset_normal import LogPlusAlphaLogNormal
from ._softplus_normal import SoftPlusNormal

NAMED_DEPTH_MODELS = {
    "LogPlusAlphaLogNormal": (LogPlusAlphaLogNormal, dict(alpha=1.0)),
    "Default": (LogPlusAlphaLogNormal, dict(alpha=1.0)),
    "SoftPlusNormal": (SoftPlusNormal, dict()),
}
