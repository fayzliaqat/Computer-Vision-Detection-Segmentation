"""Exercise actual pretrained models on the attributed Ultralytics bus image."""

from pathlib import Path
import sys
import json
import urllib.request
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.video_processor import FrameProcessor, processVideo, browserEncode

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "data" / "bus.jpg"
    if not target.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
            target,
        )
    frame = cv2.imread(str(target))
    frame = cv2.resize(frame, (480, 640))
    # Known camera translation of a still image, not a real-world motion benchmark.
    video = ROOT / "data" / "deep_smoke_test.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 640)
    )
    if not writer.isOpened():
        raise RuntimeError("Writer failed")
    for index in range(20):
        transform = np.float32([[1, 0, index * 0.7], [0, 1, 0]])
        writer.write(
            cv2.warpAffine(frame, transform, (480, 640), borderMode=cv2.BORDER_REFLECT)
        )
    writer.release()
    browserEncode(video)
    results = {}
    for mode in ("detection", "segmentation"):
        run = processVideo(
            video, ROOT / "outputs" / f"yolo_{mode}", mode=mode, line=0.5
        )
        engine = FrameProcessor(mode=mode, tracking=False)
        stages, row, detections = engine.process(frame, 1, 1)
        assert detections, "Expected actual detections on the bus photograph"
        if mode == "segmentation":
            assert any("contour" in d for d in detections), "No instance masks returned"
        cv2.imwrite(
            str(ROOT / "docs" / "demo" / f"representative_{mode}.png"),
            stages["tracked"],
        )
        results[mode] = {
            "summary": run["summary"],
            "image_class_counts": json.loads(row["class_counts"]),
            "mask_count": sum("contour" in d for d in detections),
        }
    (ROOT / "outputs" / "yolo_verification.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
