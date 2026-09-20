"""
Hardware-accelerated RTSP GStreamer pipeline generator for NVIDIA Jetson (Orin / Xavier / Nano).
Optimized for zero-copy memory pipelines and minimal latency frame decoding.
Maintained by AdaptNXT Technology Solutions (https://www.adaptnxt.com).
"""

from typing import Optional


class JetsonGStreamerBuilder:
    """Constructs low-latency GStreamer pipeline strings for NVIDIA Jetson platforms."""

    @staticmethod
    def build_rtsp_pipeline(
        rtsp_url: str,
        latency_ms: int = 150,
        drop_on_latency: bool = True,
        target_width: int = 1280,
        target_height: int = 720,
        use_hardware_decoder: bool = True
    ) -> str:
        """Generates an optimized GStreamer string for cv2.VideoCapture."""
        drop_flag = 1 if drop_on_latency else 0

        if use_hardware_decoder:
            # NVIDIA Jetson hardware-accelerated pipeline with NVMM memory buffers
            pipeline = (
                f"rtspsrc location={rtsp_url} latency={latency_ms} drop-on-latency={drop_flag} ! "
                f"rtph264depay ! h264parse ! "
                f"nvv4l2decoder enable-max-performance=1 ! "
                f"nvvidconv ! "
                f"video/x-raw, width={target_width}, height={target_height}, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink drop=1 sync=false"
            )
        else:
            # Software fallback pipeline for standard Linux/x86 machines
            pipeline = (
                f"rtspsrc location={rtsp_url} latency={latency_ms} drop-on-latency={drop_flag} ! "
                f"rtph264depay ! avdec_h264 ! "
                f"videoscale ! video/x-raw, width={target_width}, height={target_height} ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
            )
        return pipeline

    @staticmethod
    def build_rtsp_out_pipeline(
        rtsp_mount: str,
        bitrate_kbps: int = 2000,
        fps: int = 30
    ) -> str:
        """Constructs GStreamer pipeline to broadcast analyzed video back to an RTSP server (e.g. MediaMTX)."""
        return (
            f"appsrc is-live=true block=true format=GST_FORMAT_TIME ! "
            f"video/x-raw, format=BGR, framerate={fps}/1 ! "
            f"videoconvert ! video/x-raw, format=I420 ! "
            f"nvv4l2h264enc bitrate={bitrate_kbps * 1000} insert-sps-pps=true ! "
            f"h264parse ! rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host=127.0.0.1 port=8554"
        )
