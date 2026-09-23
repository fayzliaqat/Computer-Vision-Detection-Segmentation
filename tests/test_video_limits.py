from pathlib import Path
import unittest
from unittest.mock import patch

from src.video_limits import (
    MAX_TRAFFIC_CLIP_SECONDS,
    MAX_TRAFFIC_UPLOAD_BYTES,
    trafficVideoDuration,
    validateTrafficUpload,
)

ROOT = Path(__file__).resolve().parents[1]


class TrafficUploadLimitTests(unittest.TestCase):
    def test_reads_duration_from_short_mp4(self):
        duration = trafficVideoDuration(ROOT / "data" / "sample_shapes.mp4")
        self.assertGreater(duration, 0)
        self.assertLessEqual(duration, MAX_TRAFFIC_CLIP_SECONDS)

    def test_rejects_upload_over_size_limit(self):
        with self.assertRaisesRegex(ValueError, "50 MB"):
            validateTrafficUpload(ROOT / "data" / "sample_shapes.mp4", MAX_TRAFFIC_UPLOAD_BYTES + 1)

    def test_rejects_upload_over_duration_limit(self):
        with patch("src.video_limits.trafficVideoDuration", return_value=31):
            with self.assertRaisesRegex(ValueError, "30 seconds"):
                validateTrafficUpload(ROOT / "data" / "sample_shapes.mp4", 1024)


if __name__ == "__main__":
    unittest.main()
