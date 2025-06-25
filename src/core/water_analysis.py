from functools import lru_cache
from typing import Optional

from .oceangpt_manager import MarineDiseaseGPT


@lru_cache(maxsize=128)
def analyze_water_quality(temp: Optional[float], ph: Optional[float], dissolved_oxygen: Optional[float]) -> str:
    """Simple heuristic water quality analysis."""
    issues = []
    if temp is not None and not 20 <= temp <= 30:
        issues.append("温度异常")
    if ph is not None and not 6.5 <= ph <= 8.5:
        issues.append("pH异常")
    if dissolved_oxygen is not None and dissolved_oxygen < 5:
        issues.append("溶氧不足")
    if not issues:
        return "水质参数正常"
    return ";".join(issues)

