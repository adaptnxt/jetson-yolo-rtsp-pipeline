"""
Polygonal Exclusion Zone & Restricted Area Monitoring.
Fast 2D ray-casting point-in-polygon algorithm for detecting worker/vehicle intrusion in industrial safety zones.
Maintained by AdaptNXT Technology Solutions (https://www.adaptnxt.com).
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

from .inference_engine import DetectionBox
from .exceptions import ZoneConfigurationError


@dataclass
class Point:
    x: float
    y: float


@dataclass
class ZoneViolation:

    """Represents an intrusion or exclusion breach in a defined zone."""
    zone_name: str
    zone_type: str
    detection: DetectionBox
    timestamp: float
    dwell_time_seconds: float

    def to_dict(self) -> Dict[str, any]:
        return {
            "zone_name": self.zone_name,
            "zone_type": self.zone_type,
            "class_name": self.detection.class_name,
            "confidence": self.detection.confidence,
            "ground_position": list(self.detection.bottom_center),
            "dwell_time_seconds": round(self.dwell_time_seconds, 2),
            "timestamp": self.timestamp
        }


class PolygonZone:
    """Represents a 2D polygonal region of interest on a camera frame."""

    def __init__(self, name: str, vertices: List[Tuple[float, float]], zone_type: str = "EXCLUSION_ZONE"):
        if not isinstance(vertices, (list, tuple)) or len(vertices) < 3:
            raise ZoneConfigurationError("Polygon zone requires at least 3 vertices.")
        self.name = name
        self.vertices = [Point(x=v[0], y=v[1]) for v in vertices]
        self.zone_type = zone_type


    def contains_point(self, pt: Tuple[float, float]) -> bool:
        """Determines if a 2D point (x, y) falls within the polygon using Ray Casting."""
        test_x, test_y = pt
        inside = False
        n = len(self.vertices)

        p1 = self.vertices[0]
        for i in range(1, n + 1):
            p2 = self.vertices[i % n]
            if test_y > min(p1.y, p2.y):
                if test_y <= max(p1.y, p2.y):
                    if test_x <= max(p1.x, p2.x):
                        if p1.y != p2.y:
                            x_inters = (test_y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
                        if p1.x == p2.x or test_x <= x_inters:
                            inside = not inside
            p1 = p2

        return inside


class ZoneMonitor:
    """Monitors multiple polygon zones against detections and tracks dwell time."""

    def __init__(self, zones: List[PolygonZone], min_dwell_seconds: float = 0.0):
        self.zones = zones
        self.min_dwell_seconds = min_dwell_seconds
        # Tracks { (zone_name, class_name): first_seen_timestamp }
        self._active_dwell: Dict[Tuple[str, str], float] = {}

    def evaluate(self, detections: List[DetectionBox], current_time: Optional[float] = None) -> List[ZoneViolation]:
        """Evaluates detected objects against monitored zones using ground contact points."""
        now = time.time() if current_time is None else current_time
        violations: List[ZoneViolation] = []
        current_active_keys = set()


        for detection in detections:
            ground_pt = detection.bottom_center

            for zone in self.zones:
                if zone.contains_point(ground_pt):
                    key = (zone.name, f"{detection.class_name}_{detection.class_id}")
                    current_active_keys.add(key)

                    if key not in self._active_dwell:
                        self._active_dwell[key] = now

                    dwell_duration = now - self._active_dwell[key]

                    if dwell_duration >= self.min_dwell_seconds:
                        violations.append(
                            ZoneViolation(
                                zone_name=zone.name,
                                zone_type=zone.zone_type,
                                detection=detection,
                                timestamp=now,
                                dwell_time_seconds=dwell_duration
                            )
                        )

        # Cleanup expired dwell entries
        for key in list(self._active_dwell.keys()):
            if key not in current_active_keys:
                del self._active_dwell[key]

        return violations
