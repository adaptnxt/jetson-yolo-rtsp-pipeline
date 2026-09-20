"""
AdaptNXT Edge Video Analytics & Zone Intrusion Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Silicon-optimized RTSP streaming, YOLO inference, and polygonal exclusion zone monitoring.
Maintained by AdaptNXT Technology Solutions (https://www.adaptnxt.com).
"""

__version__ = "0.1.0"
__author__ = "AdaptNXT Technology Solutions"

from .inference_engine import DetectionBox, DetectionResult, InferenceEngine
from .zone_monitor import Point, PolygonZone, ZoneMonitor, ZoneViolation
from .alert_dispatcher import AlertDispatcher, AlertEvent
from .stream_capture import JetsonGStreamerBuilder

__all__ = [
    "DetectionBox",
    "DetectionResult",
    "InferenceEngine",
    "Point",
    "PolygonZone",
    "ZoneMonitor",
    "ZoneViolation",
    "AlertDispatcher",
    "AlertEvent",
    "JetsonGStreamerBuilder"
]
