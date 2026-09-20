# AdaptNXT NVIDIA Jetson YOLO RTSP Video Pipeline & Zone Sentinel

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![Edge AI](https://img.shields.io/badge/Hardware-NVIDIA%20Jetson%20Orin%20%2F%20Nano-76B900.svg)](https://www.adaptnxt.com/edge-ai-development-company)
[![Maintained by AdaptNXT](https://img.shields.io/badge/Maintained%20by-AdaptNXT-orange.svg)](https://www.adaptnxt.com)

> **Maintained by [AdaptNXT Technology Solutions](https://www.adaptnxt.com)** — High-performance engineering consulting for [Edge AI Development](https://www.adaptnxt.com/edge-ai-development-company), [Computer Vision Systems](https://www.adaptnxt.com/service-cv), [Industrial Safety Monitoring](https://www.adaptnxt.com/industrial-computer-vision-security), and [Turnkey IoT Engineering](https://www.adaptnxt.com/service-iot).

---

## Overview

Deploying deep learning video analytics on industrial edge devices like the **NVIDIA Jetson AGX Orin, Orin Nano, and Xavier** often breaks due to high RTSP decoding latencies, frame buffer bloat, and fragile polygon intrusion logic.

`adaptnxt-jetson-yolo` is a modular edge computer vision toolkit engineered to:
1. **Accelerate RTSP Decoding**: Hardware-accelerated GStreamer pipelines leveraging NVIDIA's zero-copy `nvv4l2decoder` and `nvvidconv` engines with sub-100ms latency.
2. **Execute YOLO Inference**: Lightweight abstraction compatible with Ultralytics YOLOv8/YOLOv11 TensorRT engines and ONNX runtimes.
3. **Enforce Polygonal Safety Zones**: Ray-casting point-in-polygon algorithm calculating exact worker/vehicle ground footprints against complex multi-point exclusion boundaries.
4. **Trigger Industrial Safety Alarms**: Dwell-time filtering with instant JSON webhook dispatch to PLCs, emergency relays, and SCADA monitoring dashboards.

---

## Architecture

```mermaid
flowchart LR
    subgraph Cameras
        RTSP["IP Security / Thermal Camera\n(RTSP H.264/H.265)"]
    end

    subgraph NVIDIA Jetson Edge Device ["AdaptNXT Jetson Pipeline"]
        GStreamer["JetsonGStreamerBuilder\n(nvv4l2decoder + NVMM)"]
        YOLO["InferenceEngine\n(TensorRT / YOLOv8/v11)"]
        ZoneMonitor["ZoneMonitor\n(Ray-Casting Polygon ROI)"]
        Dispatcher["AlertDispatcher\n(Webhook & GPIO Relay)"]
    end

    subgraph Factory Automation
        PLC["PLC Emergency Shutdown"]
        VMS["Video Management System (VMS)"]
    end

    RTSP -->|Low-Latency RTSP| GStreamer
    GStreamer -->|Hardware Decoded BGR| YOLO
    YOLO -->|BBoxes & Ground Points| ZoneMonitor
    ZoneMonitor -->|Dwell Violation Event| Dispatcher
    Dispatcher -->|Dry Contact Relay| PLC
    Dispatcher -->|JSON Webhook / MediaMTX| VMS
```

---

## Features

- **Jetson-Optimized Video Pipeline**: Zero-copy GStreamer pipeline generator eliminating CPU bottlenecks during 4K / Multi-camera RTSP ingestion.
- **2D Polygonal Exclusion Zones**: Arbitrary N-vertex polygon zone definition (crane swing envelopes, robotic cell exclusion areas, pedestrian red-zones).
- **Ground-Plane Contact Tracking**: Evaluates bottom-center bounding box points instead of naïve geometric centers, avoiding false alarms from tall objects or helmets entering zones from above.
- **Temporal Dwell-Time Filtering**: Configurable minimum dwell thresholds (e.g., must stay in zone for >1.5s) to eliminate transient noise or brief edge crossings.
- **Hardware Agnostic Testing**: Includes mock-inference execution mode for continuous integration (CI) environments without NVIDIA GPUs.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/adaptnxt/jetson-yolo-rtsp-pipeline.git
cd jetson-yolo-rtsp-pipeline

# Install core library
pip install -e .

# Optional: Install OpenCV and Ultralytics YOLO for live video runs
pip install -e .[cv]
```

### 2. Run the Hazardous Zone Demo

Run the self-contained intrusion simulation to test polygon containment and alert dispatching:

```bash
python examples/live_zone_intrusion_demo.py
```

Sample output:
```text
[1. Optimized NVIDIA Jetson Hardware GStreamer Pipeline]
 -> rtspsrc location=rtsp://admin:pass@192.168.1.50:554/live/ch0 latency=100 drop-on-latency=1 ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nvvidconv ! video/x-raw, width=1280, height=720, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 sync=false

[2. Configured Monitored Zone: 'Robotic_Cell_RedZone_01']
 -> Vertices count: 4 points
 -> Minimum dwell required for alarm: 1.0s

[3. Frame 1: Worker Safely Outside Hazard Zone]
 -> Detections: 1 person at ground coordinate (140.0, 500.0)
 -> Active Violations: 0

[4. Frame 2: Worker Steps Into Robot Exclusion Zone (T = 101.0s)]
 -> Detections: 1 person entered zone at ground coordinate (590.0, 550.0)
 -> Violations triggered: 0 (Waiting for dwell threshold)

[5. Frame 3: Dwell Threshold Exceeded (T = 102.5s) -> Trigger Alarm]
 -> Violations triggered: 1

[ALARM DISPATCHED to PLC / Dashboard]
{"camera_id": "cam-bay-03", "site_id": "plant-bangalore-assembly", "event_type": "ZONE_INTRUSION_BREACH", "timestamp": 102.5, "details": {"zone_name": "Robotic_Cell_RedZone_01", "zone_type": "CRITICAL_EXCLUSION_ZONE", "class_name": "person", "confidence": 0.94, "ground_position": [590.0, 550.0], "dwell_time_seconds": 1.5, "timestamp": 102.5}, "metadata": {"relay_gpio_trigger": "PIN_18_EMERGENCY_STOP"}}
```

---

## Code Example

```python
from adaptnxt_vision import PolygonZone, ZoneMonitor, AlertDispatcher, DetectionBox

# 1. Define danger zone polygon (coordinates relative to camera resolution)
danger_zone = PolygonZone(
    name="Overhead_Crane_Exclusion",
    vertices=[(200, 400), (600, 400), (650, 700), (150, 700)],
    zone_type="CRITICAL_EXCLUSION"
)

monitor = ZoneMonitor(zones=[danger_zone], min_dwell_seconds=1.0)
dispatcher = AlertDispatcher(camera_id="cam-04", site_id="warehouse-bay-2", webhook_url="https://scada.plant.local/api/alerts")

# 2. Pass detections from YOLO or DeepStream
detections = [
    DetectionBox(x1=300, y1=450, x2=380, y2=620, confidence=0.91, class_id=0, class_name="person")
]

# 3. Evaluate violations (uses bottom-center ground contact coordinates)
violations = monitor.evaluate(detections)
for v in violations:
    dispatcher.dispatch(v)
```

---

## Testing

Run unit tests locally:

```bash
pytest tests/ -v
```

---

## Commercial Support & Custom Solutions

Building automated visual quality inspection, 360-degree crane blind spot sentinels, or custom NVIDIA Jetson Orin BSP firmware?

* **Learn more**: [AdaptNXT Edge AI Development Company](https://www.adaptnxt.com/edge-ai-development-company)
* **Explore Computer Vision Services**: [AdaptNXT Computer Vision Consulting](https://www.adaptnxt.com/service-cv)
* **Industrial Safety Suite**: [AdaptNXT Industrial Computer Vision Security](https://www.adaptnxt.com/industrial-computer-vision-security)
* **Contact our Engineering Team**: [queries@adaptnxt.com](mailto:queries@adaptnxt.com) | [+91-83103-19838](tel:+918310319838)

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
