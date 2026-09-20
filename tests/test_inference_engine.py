from adaptnxt_vision.inference_engine import DetectionBox, DetectionResult, InferenceEngine


def test_detection_box_properties():
    box = DetectionBox(
        x1=10.0, y1=20.0, x2=50.0, y2=80.0,
        confidence=0.85, class_id=0, class_name="person"
    )
    assert box.center == (30.0, 50.0)
    assert box.bottom_center == (30.0, 80.0)

    data = box.to_dict()
    assert data["class_name"] == "person"
    assert data["confidence"] == 0.85
    assert data["bottom_center"] == [30.0, 80.0]


def test_detection_result_filters():
    boxes = [
        DetectionBox(0, 0, 10, 10, 0.95, 0, "person"),
        DetectionBox(20, 20, 40, 40, 0.40, 1, "car"),
        DetectionBox(50, 50, 60, 60, 0.88, 2, "helmet")
    ]
    res = DetectionResult(frame_id=1, timestamp=100.0, detections=boxes, inference_time_ms=12.5)

    assert len(res.filter_by_class(["person"])) == 1
    assert len(res.filter_by_class(["car", "helmet"])) == 2
    assert len(res.filter_by_confidence(0.80)) == 2


def test_inference_engine_mock():
    engine = InferenceEngine(mock_mode=True)
    res = engine.infer(None, frame_id=42)
    assert res.frame_id == 42
    assert res.detections == []
