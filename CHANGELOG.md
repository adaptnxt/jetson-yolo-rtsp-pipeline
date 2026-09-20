# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-20

### Added
- Low-latency hardware GStreamer pipeline generator for NVIDIA Jetson (`JetsonGStreamerBuilder`) using `nvv4l2decoder` and NVMM zero-copy memory.
- Object detection bounding box structures (`DetectionBox`, `DetectionResult`) tracking bottom-center ground contact coordinates.
- Ultralytics YOLO inference wrapper (`InferenceEngine`) with TensorRT and mock execution support.
- 2D ray-casting polygonal exclusion zone monitoring (`PolygonZone`, `ZoneMonitor`) with temporal dwell-time filtering.
- Automated alert notification dispatcher (`AlertDispatcher`) with HTTP webhook transmission for PLC and SCADA integration.
- Command-line interface (`adaptnxt-vision`) for generating GStreamer strings and running simulated sentinel demos.
- Complete unit test suite with 100% pass rate in CI without requiring physical NVIDIA GPUs.
