"""
Alert Dispatcher & Webhook Notification Engine for Computer Vision Events.
Maintained by AdaptNXT Technology Solutions (https://www.adaptnxt.com).
"""

from dataclasses import dataclass
import json
import logging
from typing import Callable, Dict, List, Optional
import urllib.request
import urllib.error

from .zone_monitor import ZoneViolation

logger = logging.getLogger("adaptnxt_vision.alerts")


@dataclass
class AlertEvent:
    camera_id: str
    site_id: str
    event_type: str
    violation: ZoneViolation
    metadata: Optional[Dict[str, any]] = None

    def to_dict(self) -> Dict[str, any]:
        return {
            "camera_id": self.camera_id,
            "site_id": self.site_id,
            "event_type": self.event_type,
            "timestamp": self.violation.timestamp,
            "details": self.violation.to_dict(),
            "metadata": self.metadata or {}
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AlertDispatcher:
    """Dispatches computer vision violation events to webhooks or alert handlers."""

    def __init__(
        self,
        camera_id: str = "cam-01",
        site_id: str = "factory-floor",
        webhook_url: Optional[str] = None
    ):
        self.camera_id = camera_id
        self.site_id = site_id
        self.webhook_url = webhook_url
        self._handlers: List[Callable[[AlertEvent], None]] = []

    def add_handler(self, callback: Callable[[AlertEvent], None]) -> None:
        """Registers a custom Python callback function for violations."""
        self._handlers.append(callback)

    def dispatch(self, violation: ZoneViolation, metadata: Optional[Dict[str, any]] = None) -> AlertEvent:
        """Dispatches an alert event across registered handlers and optional webhook."""
        event = AlertEvent(
            camera_id=self.camera_id,
            site_id=self.site_id,
            event_type="ZONE_INTRUSION_BREACH",
            violation=violation,
            metadata=metadata
        )

        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error("Error executing alert handler: %s", e)

        if self.webhook_url:
            self._send_webhook(event)

        return event

    def _send_webhook(self, event: AlertEvent) -> bool:
        """Transmits JSON event to HTTP endpoint."""
        try:
            payload = event.to_json().encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "AdaptNXT-Vision-Sentinel/1.0"}
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status in (200, 201, 204)
        except Exception as e:
            logger.warning("Webhook delivery failed to %s: %s", self.webhook_url, e)
            return False
