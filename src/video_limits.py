"""Resource limits and metadata checks for uploaded traffic clips."""

from pathlib import Path

import cv2

MAX_TRAFFIC_CLIP_SECONDS = 30
MAX_TRAFFIC_UPLOAD_BYTES = 50 * 1024 * 1024


def validateTrafficUpload(path: Path, sizeBytes: int) -> float:
    """Enforce the public traffic demo's upload size and duration limits."""
    if sizeBytes > MAX_TRAFFIC_UPLOAD_BYTES:
        raise ValueError("Traffic uploads are limited to 50 MB. Choose a smaller MP4.")

    duration = trafficVideoDuration(path)
    if duration > MAX_TRAFFIC_CLIP_SECONDS:
        raise ValueError(
            "Traffic uploads are limited to 30 seconds. Trim the MP4 and try again."
        )
    return duration


def trafficVideoDuration(path: Path) -> float:
    """Return the duration from video metadata, rejecting unreadable files."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(
                "Could not read this MP4. Try a valid H.264 encoded video."
            )
        frameCount = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()

    if frameCount <= 0 or fps <= 0:
        raise ValueError("Could not read the MP4 duration. Try a valid MP4 file.")
    return frameCount / fps
