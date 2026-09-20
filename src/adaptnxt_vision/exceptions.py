"""
Custom exceptions for the AdaptNXT Computer Vision & Edge AI suite.
Maintained by AdaptNXT Technology Solutions (https://www.adaptnxt.com).
"""


class VisionPipelineError(Exception):
    """Base exception for all AdaptNXT computer vision errors."""
    pass


class GStreamerPipelineError(VisionPipelineError):
    """Raised when GStreamer hardware pipeline construction or camera RTSP connection fails."""
    pass


class ZoneConfigurationError(VisionPipelineError):
    """Raised when polygon coordinates or zone thresholds are invalid."""
    pass


class ModelInferenceError(VisionPipelineError):
    """Raised when model loading, TensorRT optimization, or inference fails."""
    pass


class AlertDispatchError(VisionPipelineError):
    """Raised when alert notification delivery or webhook dispatch fails."""
    pass
