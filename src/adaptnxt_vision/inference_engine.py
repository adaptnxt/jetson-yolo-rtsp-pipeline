"""
Edge AI Inference Engine wrapper for YOLO object detection models.
Maintained by AdaptNXT Technology Solutions (https://www.adaptnxt.com).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DetectionBox:
    """Represents an object bounding box and classification."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        """Calculates bbox geometric center."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Calculates bottom center point (critical for ground-plane footprint in camera surveillance)."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)],
            "confidence": round(self.confidence, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bottom_center": [round(self.bottom_center[0], 1), round(self.bottom_center[1], 1)]
        }


@dataclass
class DetectionResult:
    """Contains all detections from an ingested video frame."""
    frame_id: int
    timestamp: float
    detections: List[DetectionBox]
    inference_time_ms: float

    def filter_by_class(self, class_names: List[str]) -> List[DetectionBox]:
        return [d for d in self.detections if d.class_name in class_names]

    def filter_by_confidence(self, min_conf: float) -> List[DetectionBox]:
        return [d for d in self.detections if d.confidence >= min_conf]


class InferenceEngine:
    """Wrapper that executes YOLO models via TensorRT, ONNX, or Mock mode."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
        mock_mode: bool = False
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or []
        self.mock_mode = mock_mode or (model_path is None)
        self._model = None

        if not self.mock_mode:
            self._load_yolo()

    def _load_yolo(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        except ImportError:
            self.mock_mode = True

    def infer(self, frame: Any, frame_id: int = 0) -> DetectionResult:
        """Executes inference on an image frame."""
        import time
        start_t = time.time()

        if self.mock_mode:
            # Deterministic empty or passthrough result in mock mode
            return DetectionResult(
                frame_id=frame_id,
                timestamp=start_t,
                detections=[],
                inference_time_ms=1.5
            )

        results = self._model(frame, conf=self.confidence_threshold, verbose=False)
        boxes: List[DetectionBox] = []

        if results and len(results) > 0:
            r = results[0]
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                coords = box.xyxy[0].tolist()
                name = r.names.get(cls_id, f"class_{cls_id}")

                if self.target_classes and name not in self.target_classes:
                    continue

                boxes.append(
                    DetectionBox(
                        x1=coords[0],
                        y1=coords[1],
                        x2=coords[2],
                        y2=coords[3],
                        confidence=conf,
                        class_id=cls_id,
                        class_name=name
                    )
                )

        duration_ms = (time.time() - start_t) * 1000.0
        return DetectionResult(
            frame_id=frame_id,
            timestamp=start_t,
            detections=boxes,
            inference_time_ms=round(duration_ms, 2)
        )
