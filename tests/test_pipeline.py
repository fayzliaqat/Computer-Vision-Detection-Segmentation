import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
from src.preprocessing import VisionConfig, preprocessFrame
from src.segmentation import extractContours
from src.tracking import CentroidTracker, SceneAnalytics
from src.video_processor import processVideo, validateVideo, FrameProcessor

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def test_shapes_and_area_bounds(self):
        frame = np.full((300, 600, 3), 240, np.uint8)
        cv2.circle(frame, (90, 140), 35, (30, 30, 30), -1)
        cv2.rectangle(frame, (220, 100), (290, 170), (30, 30, 30), -1)
        cv2.fillPoly(
            frame, [np.array([[420, 90], [380, 170], [460, 170]])], (30, 30, 30)
        )
        config = VisionConfig()
        stages = preprocessFrame(frame, config)
        detections, raw = extractContours(stages["mask"], config)
        self.assertEqual(
            {d["label"] for d in detections}, {"Circle", "Rectangle", "Triangle"}
        )
        self.assertEqual(raw, 3)
        self.assertEqual(set(np.unique(stages["threshold"])), {0, 255})
        self.assertFalse(np.array_equal(stages["grayscale"], stages["blurred"]))
        config.minArea, config.maxArea = 10000, 20000
        self.assertEqual(extractContours(stages["mask"], config)[0], [])

    def test_invalid_configuration(self):
        for config in [
            VisionConfig(blockSize=2),
            VisionConfig(minArea=300, maxArea=200),
            VisionConfig(gaussianKernel=4),
            VisionConfig(thresholdC=float("nan")),
        ]:
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            SceneAnalytics((0.9, 0.1, 0.2, 0.8))

    def test_tracker_persistence_one_to_one_and_expiry(self):
        tracker = CentroidTracker(20, 1)
        first = tracker.update([{"center": (10, 10)}, {"center": (30, 10)}])
        ids = [d["track_id"] for d in first]
        second = tracker.update([{"center": (15, 10)}, {"center": (28, 10)}])
        self.assertEqual([d["track_id"] for d in second], ids)
        third = tracker.update([{"center": (16, 10)}, {"center": (17, 10)}])
        self.assertEqual(len({d["track_id"] for d in third}), 2)
        tracker.update([])
        tracker.update([])
        self.assertFalse(tracker.tracks)
        self.assertGreater(
            tracker.update([{"center": (10, 10)}])[0]["track_id"], max(ids)
        )

    def test_crossing_once_and_roi(self):
        analytics = SceneAnalytics((0.2, 0.2, 0.8, 0.8), 0.5)
        for x in [30, 48, 50, 52, 70, 30, 70]:
            counts = analytics.update(
                [{"center": (x, 50), "track_id": 1, "label": "Circle"}], 100, 100
            )
            self.assertEqual(counts, {"Circle": 1})
        self.assertEqual((analytics.entered, analytics.exited), (1, 0))
        self.assertEqual(
            analytics.update([{"center": (99, 99), "label": "Circle"}], 100, 100), {}
        )
        self.assertEqual(
            analytics.update([{"center": (50, 50), "label": "Circle"}], 100, 100),
            {"Circle": 1},
        )

    def test_bounded_trails(self):
        tracker = CentroidTracker()
        for x in range(100):
            tracker.update([{"center": (x, 10)}])
        self.assertEqual(len(tracker.tracks[1]["history"]), 24)

    def test_upload_bytes_roundtrip_and_metrics(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            folder = Path(directory)
            upload = folder / "uploaded.mp4"
            upload.write_bytes((ROOT / "data" / "sample_shapes.mp4").read_bytes())
            run = processVideo(upload, folder / "result", maxFrames=12)
            df = pd.read_csv(folder / "result" / "frame_metrics.csv")
            self.assertEqual(len(df), 12)
            self.assertTrue((df.processing_time_ms > 0).all())
            np.testing.assert_allclose(df.fps, 1000 / df.processing_time_ms)
            self.assertEqual(validateVideo(folder / "result" / run["video"]), 12)
            self.assertEqual(run["summary"]["unique_tracks"], 3)

    def test_missing_input_and_yolo_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            with self.assertRaises(ValueError):
                processVideo(ROOT / "absent.mp4", directory)
        with patch(
            "src.yolo_detector.YoloDetector", side_effect=RuntimeError("offline")
        ):
            with self.assertRaises(RuntimeError):
                FrameProcessor(mode="detection")
        engine = FrameProcessor()
        _, row, _ = engine.process(np.full((200, 300, 3), 240, np.uint8), 1, 30)
        self.assertEqual(row["objects_detected"], 0)

    def test_artifact_integrity(self):
        df = pd.read_csv(ROOT / "outputs" / "frame_metrics.csv")
        self.assertEqual(len(df), 240)
        self.assertEqual(
            validateVideo(ROOT / "outputs" / "videos" / "opencv_segmented_output.mp4"),
            240,
        )
        for name in [
            "pipeline_stages",
            "processing_time_by_frame",
            "fps_by_frame",
            "object_count_by_frame",
            "shape_counts_by_frame",
        ]:
            self.assertTrue((ROOT / "outputs" / "charts" / f"{name}.png").exists())
        evaluation = json.loads(
            (ROOT / "outputs" / "synthetic_evaluation.json").read_text()
        )
        self.assertEqual(evaluation["matched"], 865)
        self.assertEqual(evaluation["count_mae"], 0)

    def test_camera_unavailable(self):
        import realtime_demo

        with (
            patch("sys.argv", ["realtime_demo.py"]),
            patch("realtime_demo.cv2.VideoCapture") as mock,
        ):
            mock.return_value.isOpened.return_value = False
            self.assertEqual(realtime_demo.main(), 1)
            mock.return_value.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
