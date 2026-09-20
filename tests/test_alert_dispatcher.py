import json
from adaptnxt_vision.alert_dispatcher import AlertDispatcher
from adaptnxt_vision.zone_monitor import ZoneViolation
from adaptnxt_vision.inference_engine import DetectionBox


def test_alert_dispatcher_handlers():
    dispatcher = AlertDispatcher(camera_id="cam-01", site_id="warehouse-zone")

    events_received = []
    dispatcher.add_handler(lambda evt: events_received.append(evt))

    box = DetectionBox(10, 10, 20, 20, 0.95, 0, "person")
    violation = ZoneViolation(
        zone_name="Crane_RedZone",
        zone_type="CRITICAL",
        detection=box,
        timestamp=123456.0,
        dwell_time_seconds=3.2
    )

    event = dispatcher.dispatch(violation, metadata={"severity": "HIGH"})

    assert len(events_received) == 1
    assert event.camera_id == "cam-01"
    assert event.site_id == "warehouse-zone"
    assert event.metadata["severity"] == "HIGH"

    # Verify JSON serialization
    raw_json = event.to_json()
    parsed = json.loads(raw_json)
    assert parsed["camera_id"] == "cam-01"
    assert parsed["details"]["zone_name"] == "Crane_RedZone"
    assert parsed["details"]["dwell_time_seconds"] == 3.2
