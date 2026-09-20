"""
Command-Line Interface for the AdaptNXT Computer Vision & Jetson Sentinel Pipeline.
Maintained by AdaptNXT Technology Solutions (https://www.adaptnxt.com).
"""

import argparse
import sys

from . import __version__
from .stream_capture import JetsonGStreamerBuilder


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="adaptnxt-vision",
        description="AdaptNXT Edge Video Analytics & Jetson Zone Sentinel CLI"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: gstreamer
    gst_parser = subparsers.add_parser("pipeline", help="Generate optimized Jetson GStreamer pipeline string")
    gst_parser.add_argument("--rtsp", required=True, help="RTSP stream URL (e.g. rtsp://192.168.1.50:554/live)")
    gst_parser.add_argument("--latency", type=int, default=150, help="Jitter buffer latency in ms (default: 150)")
    gst_parser.add_argument("--width", type=int, default=1280, help="Output frame width (default: 1280)")
    gst_parser.add_argument("--height", type=int, default=720, help="Output frame height (default: 720)")
    gst_parser.add_argument("--software", action="store_true", help="Use software decoder instead of nvv4l2decoder")

    # Subcommand: demo
    subparsers.add_parser("demo", help="Run simulated hazardous zone intrusion sentinel")

    args = parser.parse_args()

    if args.command == "pipeline":
        pipeline_str = JetsonGStreamerBuilder.build_rtsp_pipeline(
            rtsp_url=args.rtsp,
            latency_ms=args.latency,
            target_width=args.width,
            target_height=args.height,
            use_hardware_decoder=not args.software
        )
        print("\nOptimized Jetson GStreamer Pipeline String:")
        print(pipeline_str)
        print("\nPass this string directly to cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)\n")
    elif args.command == "demo":
        from adaptnxt_vision.demo import run_demo
        run_demo()
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
