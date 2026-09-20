"""
End-to-End Demonstration: Polygonal Hazard Zone Intrusion Detection for NVIDIA Jetson & Edge AI.
Demonstrates tracking worker/vehicle coordinates against restricted polygonal boundaries.
Maintained by AdaptNXT Technology Solutions (https://www.adaptnxt.com).
"""

import sys
import os
import time

# Allow direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from adaptnxt_vision import (
    DetectionBox,
    PolygonZone,
    ZoneMonitor,
    AlertDispatcher,
    JetsonGStreamerBuilder
)


def run_demo():
    print("================================================================")
    print(" AdaptNXT Edge Computer Vision: Hazardous Zone Intrusion Sentinel")
    print(" https://www.adaptnxt.com/edge-ai-development-company")
    print("================================================================\n")

    # Step 1: Demonstrate Jetson GStreamer pipeline string creation
    rtsp_url = "rtsp://admin:pass@192.168.1.50:554/live/ch0"
    gstreamer_str = JetsonGStreamerBuilder.build_rtsp_pipeline(rtsp_url=rtsp_url, latency_ms=100)
    print("[1. Optimized NVIDIA Jetson Hardware GStreamer Pipeline]")
    print(f" -> {gstreamer_str}\n")

    # Step 2: Define a Polygonal Exclusion Red-Zone (e.g. Robotic Arm Working Envelope)
    # Coordinates in 1280x720 frame:
    crane_envelope = PolygonZone(
        name="Robotic_Cell_RedZone_01",
        vertices=[
            (400.0, 300.0),
            (800.0, 300.0),
            (850.0, 650.0),
            (350.0, 650.0)
        ],
        zone_type="CRITICAL_EXCLUSION_ZONE"
    )

    monitor = ZoneMonitor(zones=[crane_envelope], min_dwell_seconds=1.0)
    dispatcher = AlertDispatcher(camera_id="cam-bay-03", site_id="plant-bangalore-assembly")

    captured_alerts = []
    dispatcher.add_handler(lambda event: captured_alerts.append(event))

    print(f"[2. Configured Monitored Zone: '{crane_envelope.name}']")
    print(f" -> Vertices count: {len(crane_envelope.vertices)} points")
    print(f" -> Minimum dwell required for alarm: {monitor.min_dwell_seconds}s\n")

    # Step 3: Simulation Frame 1 — Worker outside the danger zone
    print("[3. Frame 1: Worker Safely Outside Hazard Zone]")
    box_outside = DetectionBox(
        x1=100.0, y1=300.0, x2=180.0, y2=500.0,  # Bottom center is (140, 500) -> Outside
        confidence=0.92, class_id=0, class_name="person"
    )
    violations = monitor.evaluate([box_outside], current_time=100.0)
    print(f" -> Detections: 1 person at ground coordinate ({box_outside.bottom_center[0]}, {box_outside.bottom_center[1]})")
    print(f" -> Active Violations: {len(violations)}\n")

    # Step 4: Simulation Frame 2 — Worker steps into the danger zone (T = 101.0s)
    print("[4. Frame 2: Worker Steps Into Robot Exclusion Zone (T = 101.0s)]")
    box_inside = DetectionBox(
        x1=550.0, y1=350.0, x2=630.0, y2=550.0,  # Bottom center is (590, 550) -> Inside
        confidence=0.94, class_id=0, class_name="person"
    )
    # First instant of entry (dwell = 0.0s < 1.0s min dwell)
    violations = monitor.evaluate([box_inside], current_time=101.0)
    print(f" -> Detections: 1 person entered zone at ground coordinate {box_inside.bottom_center}")
    print(f" -> Violations triggered: {len(violations)} (Waiting for dwell threshold)")

    # Step 5: Simulation Frame 3 — Worker remains in danger zone for 1.5 seconds (T = 102.5s)
    print("\n[5. Frame 3: Dwell Threshold Exceeded (T = 102.5s) -> Trigger Alarm]")
    violations = monitor.evaluate([box_inside], current_time=102.5)
    print(f" -> Violations triggered: {len(violations)}")

    for v in violations:
        event = dispatcher.dispatch(v, metadata={"relay_gpio_trigger": "PIN_18_EMERGENCY_STOP"})
        print(f"\n[ALARM DISPATCHED to PLC / Dashboard]")
        print(event.to_json())

    print("\nIntrusion surveillance simulation completed successfully!")


if __name__ == "__main__":
    run_demo()
