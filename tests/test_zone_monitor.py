import pytest
from adaptnxt_vision.zone_monitor import PolygonZone, ZoneMonitor
from adaptnxt_vision.inference_engine import DetectionBox
from adaptnxt_vision.exceptions import ZoneConfigurationError


@pytest.fixture
def square_zone():
    # 100x100 square from (100, 100) to (200, 200)
    return PolygonZone(
        name="test_box",
        vertices=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]
    )


def test_zone_configuration_error():
    with pytest.raises(ZoneConfigurationError):
        PolygonZone(name="bad_zone", vertices=[(0.0, 0.0), (1.0, 1.0)])



def test_polygon_contains_point(square_zone):
    # Inside point
    assert square_zone.contains_point((150.0, 150.0)) is True
    # Outside points
    assert square_zone.contains_point((50.0, 150.0)) is False
    assert square_zone.contains_point((250.0, 150.0)) is False
    assert square_zone.contains_point((150.0, 250.0)) is False


def test_zone_monitor_violation(square_zone):
    monitor = ZoneMonitor(zones=[square_zone], min_dwell_seconds=0.0)

    # Box whose bottom center is at (150, 150) -> Inside
    box_inside = DetectionBox(
        x1=130.0, y1=100.0, x2=170.0, y2=150.0,
        confidence=0.9, class_id=0, class_name="person"
    )

    violations = monitor.evaluate([box_inside], current_time=1.0)
    assert len(violations) == 1
    assert violations[0].zone_name == "test_box"
    assert violations[0].detection.class_name == "person"


def test_dwell_time_filtering(square_zone):
    monitor = ZoneMonitor(zones=[square_zone], min_dwell_seconds=2.0)

    box_inside = DetectionBox(
        x1=130.0, y1=100.0, x2=170.0, y2=150.0,
        confidence=0.9, class_id=0, class_name="person"
    )

    # At t=0s, dwell=0s (< 2.0s min dwell) -> No violation yet
    violations_t0 = monitor.evaluate([box_inside], current_time=0.0)
    assert len(violations_t0) == 0

    # At t=1.0s, dwell=1.0s (< 2.0s min dwell) -> No violation yet
    violations_t1 = monitor.evaluate([box_inside], current_time=1.0)
    assert len(violations_t1) == 0

    # At t=2.5s, dwell=2.5s (>= 2.0s min dwell) -> Triggers violation
    violations_t2 = monitor.evaluate([box_inside], current_time=2.5)
    assert len(violations_t2) == 1
    assert violations_t2[0].dwell_time_seconds == 2.5
