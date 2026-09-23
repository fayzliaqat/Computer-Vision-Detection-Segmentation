import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
from src.tracking import SceneAnalytics
from src.traffic import VEHICLE_CLASSES, TRAFFIC_FOOTER, vehicleCounts, summarizeTraffic
from src.video_processor import FrameProcessor, processVideo

ROOT = Path(__file__).resolve().parents[1]


class TrafficTests(unittest.TestCase):
    def test_horizontal_directions_jitter_and_once_per_id(self):
        analytics = SceneAnalytics(line=0.5, lineOrientation="horizontal")
        for y in [30, 49, 51, 55, 49, 30, 70]:
            analytics.update(
                [{"center": (50, y), "track_id": 1, "label": "car"}], 100, 100
            )
        for y in [70, 50, 45]:
            analytics.update(
                [{"center": (60, y), "track_id": 2, "label": "truck"}], 100, 100
            )
        self.assertEqual((analytics.entered, analytics.exited), (1, 1))
        self.assertEqual([e["direction"] for e in analytics.events], ["A", "B"])
        self.assertEqual(len(analytics.counted), 2)

    def test_no_id_no_crossing_and_invalid_orientation(self):
        analytics = SceneAnalytics(line=0.5, lineOrientation="horizontal")
        for y in [20, 80]:
            analytics.update([{"center": (50, y), "label": "car"}], 100, 100)
        self.assertEqual(analytics.events, [])
        with self.assertRaises(ValueError):
            SceneAnalytics(lineOrientation="diagonal")
        with self.assertRaises(ValueError):
            SceneAnalytics(trailLength=1000)
        with self.assertRaises(ValueError):
            FrameProcessor(mode="classical", traffic=True)
        with self.assertRaises(ValueError):
            FrameProcessor(mode="detection", traffic=True, vehicleClasses=[])

    def test_vehicle_metrics_exclude_unrelated_classes(self):
        counts = vehicleCounts(
            [
                {"label": "person"},
                {"label": "car"},
                {"label": "car"},
                {"label": "truck"},
            ]
        )
        self.assertEqual(set(counts), set(VEHICLE_CLASSES))
        self.assertEqual(sum(counts.values()), 3)
        rows = pd.DataFrame(
            [
                dict(
                    active_vehicles=3,
                    direction_a=1,
                    direction_b=2,
                    total_crossings=3,
                    inference_time_ms=10,
                    vehicle_class_counts=json.dumps(counts),
                )
            ]
        )
        result = summarizeTraffic(rows)
        self.assertEqual(result["total_crossings"], 3)
        self.assertEqual(result["vehicle_classes"]["car"]["mean_visible"], 2)

    def test_filtered_engine_events_and_video_export_without_weights(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        calls = [0]

        def predict(frame):
            y = [70, 90, 120][calls[0]]
            calls[0] += 1
            return [
                dict(
                    label="car", center=(80, y), bbox=(65, y - 10, 30, 20), track_id=7
                ),
                dict(
                    label="person",
                    center=(160, y),
                    bbox=(140, y - 10, 40, 20),
                    track_id=8,
                ),
            ], 2.0

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            folder = Path(directory)
            source = folder / "test.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 200)
            )
            for _ in range(3):
                writer.write(np.full((200, 320, 3), 200, np.uint8))
            writer.release()
            with patch("src.yolo_detector.YoloDetector") as model:
                model.return_value.detect.side_effect = predict
                run = processVideo(
                    source,
                    folder / "result",
                    mode="detection",
                    traffic=True,
                    line=0.5,
                    lineOrientation="horizontal",
                )
                self.assertEqual(model.call_args.kwargs["classNames"], VEHICLE_CLASSES)
            self.assertEqual(run["summary"]["total_crossings"], 1)
            self.assertEqual(run["summary"]["direction_a"], 1)
            self.assertEqual(run["summary"]["peak_active_vehicles"], 1)
            events = pd.read_csv(folder / "result/crossing_events.csv")
            self.assertEqual(events["class"].tolist(), ["car"])
            self.assertAlmostEqual(events.timestamp_seconds.iloc[0], 0.08)
            capture = cv2.VideoCapture(str(folder / "result" / run["video"]))
            self.assertEqual(
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 200 + TRAFFIC_FOOTER
            )
            capture.release()

    def test_saved_traffic_counts_are_auditable(self):
        folder = ROOT / "outputs/traffic"
        if not (folder / "run.json").exists():
            self.skipTest("No committed traffic run")
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        events = pd.read_csv(folder / "crossing_events.csv")
        frames = pd.read_csv(folder / "frame_metrics.csv")
        self.assertEqual(len(events), run["summary"]["total_crossings"])
        self.assertEqual(events.track_id.nunique(), len(events))
        self.assertEqual(len(frames), run["summary"]["total_frames"])
        self.assertTrue(set(events["class"]) <= set(VEHICLE_CLASSES))
        self.assertEqual(
            int((events.direction == "A").sum()), run["summary"]["direction_a"]
        )


if __name__ == "__main__":
    unittest.main()
