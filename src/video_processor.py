"""Shared image/video processing, annotation and bounded representative artifacts."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import platform
import shutil
import subprocess
import time
import cv2
import numpy as np
import pandas as pd
from .preprocessing import VisionConfig, preprocessFrame
from .segmentation import extractContours
from .tracking import CentroidTracker, SceneAnalytics
from .metrics import summarizeMetrics, createCharts, pipelineFigure

COLORS = {
    "Circle": (176, 201, 54),
    "Rectangle": (115, 195, 255),
    "Triangle": (247, 172, 113),
    "Other": (189, 150, 220),
}


def annotate(frame, detections, analytics, trails=True, masks=False):
    output = frame.copy()
    height, width = frame.shape[:2]
    for det in detections:
        color = COLORS.get(det["label"], (176, 201, 54))
        contour = det.get("contour")
        if contour is not None:
            if masks:
                layer = output.copy()
                cv2.drawContours(layer, [contour], -1, color, -1)
                output = cv2.addWeighted(output, 0.7, layer, 0.3, 0)
            cv2.drawContours(output, [contour], -1, color, 2)
        x, y, w, h = det["bbox"]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        trackId = det.get("track_id")
        label = (f"#{trackId} " if trackId is not None else "") + det["label"]
        if "confidence" in det:
            label += f" {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ty = max(th + 5, y - 6)
        cv2.rectangle(
            output,
            (x, ty - th - 4),
            (min(width - 1, x + tw + 5), ty + 3),
            (25, 32, 42),
            -1,
        )
        cv2.putText(
            output,
            label,
            (x + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        if trails and trackId in analytics.histories:
            points = np.array(analytics.histories[trackId], np.int32)
            if len(points) > 1:
                cv2.polylines(output, [points], False, color, 2, cv2.LINE_AA)
    if analytics.roi:
        l, t, r, b = analytics.roi
        cv2.rectangle(
            output,
            (int(l * width), int(t * height)),
            (int(r * width), int(b * height)),
            (215, 185, 95),
            1,
        )
    if analytics.line is not None:
        x = int(analytics.line * width)
        cv2.line(output, (x, 0), (x, height), (100, 200, 255), 2)
    return output


def drawHud(frame, row, mode):
    # A compact footer leaves most of the scene visible.
    output = frame.copy()
    height, width = output.shape[:2]
    scale = max(0.3, min(0.5, width / 1500))
    lines = [
        f"{mode.upper()} | FRAME {row['frame_number']} | OBJECTS {row['objects_detected']} | TRACKS {row['active_tracks']}",
        f"{row['processing_time_ms']:.2f} ms | {row['fps']:.1f} processing FPS | ROI {row['roi_objects']} | IN {row['entered']} OUT {row['exited']}",
        "  ".join(f"{k}: {v}" for k, v in json.loads(row["class_counts"]).items())
        or "No targets",
    ]
    cv2.rectangle(output, (0, max(0, height - 66)), (width, height), (27, 22, 13), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line[:145],
            (10, height - 48 + index * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (229, 237, 245),
            1,
            cv2.LINE_AA,
        )
    return output


class FrameProcessor:
    def __init__(
        self,
        config=None,
        mode="classical",
        confidence=0.25,
        tracking=True,
        roi=None,
        line=None,
        trails=True,
    ):
        if mode not in ("classical", "detection", "segmentation"):
            raise ValueError("Unknown vision mode")
        self.config = (config or VisionConfig()).validate()
        self.mode, self.trails = mode, trails
        self.analytics = SceneAnalytics(roi, line)
        self.tracker = CentroidTracker(
            self.config.maxDistance, self.config.maxDisappeared
        )
        self.seenIds = set()
        self.deep = None
        if mode != "classical":
            from .yolo_detector import YoloDetector

            self.deep = YoloDetector(mode == "segmentation", confidence, tracking)

    def process(self, frame, frameNumber, sourceFps):
        start = time.perf_counter()
        inferenceMs = None
        if self.mode == "classical":
            stages = preprocessFrame(frame, self.config)
            detections, rawCount = extractContours(stages["mask"], self.config)
            self.tracker.update(detections)
        else:
            stages = {"original": frame.copy()}
            detections, inferenceMs = self.deep.detect(frame)
            rawCount = None
        height, width = frame.shape[:2]
        roiCounts = self.analytics.update(detections, width, height)
        currentIds = {
            d["track_id"] for d in detections if d.get("track_id") is not None
        }
        self.seenIds.update(currentIds)
        counts = Counter(d["label"] for d in detections)
        untracked = [
            {k: v for k, v in d.items() if k != "track_id"} for d in detections
        ]
        stages["segmented"] = annotate(
            frame, untracked, self.analytics, False, self.mode == "segmentation"
        )
        tracked = annotate(
            frame, detections, self.analytics, self.trails, self.mode == "segmentation"
        )
        elapsed = max((time.perf_counter() - start) * 1000, 1e-9)
        row = {
            "frame_number": frameNumber,
            "timestamp_seconds": (frameNumber - 1) / sourceFps,
            "objects_detected": len(detections),
            "active_tracks": len(currentIds),
            "processing_time_ms": elapsed,
            "fps": 1000 / elapsed,
            "contour_count_raw": rawCount,
            "contour_count_filtered": len(detections)
            if self.mode == "classical"
            else None,
            "inference_time_ms": inferenceMs,
            "class_counts": json.dumps(dict(counts)),
            "roi_objects": sum(roiCounts.values()),
            "roi_class_counts": json.dumps(roiCounts),
            "entered": self.analytics.entered,
            "exited": self.analytics.exited,
        }
        for label in ("Circle", "Rectangle", "Triangle", "Other"):
            row[label.lower() + "_count"] = counts.get(label, 0)
        stages["tracked"] = drawHud(tracked, row, self.mode)
        return stages, row, detections


def browserEncode(path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return "mp4v (install FFmpeg for browser-compatible H.264)"
    converted = path.with_name(path.stem + "_h264.mp4")
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(converted),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "FFmpeg could not encode the annotated video: " + result.stderr[-500:]
        )
    converted.replace(path)
    return "H.264 / yuv420p"


def validateVideo(path, expected=None):
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    if count == 0 or (expected is not None and count != expected):
        raise RuntimeError(
            f"Video verification failed: decoded {count}, expected {expected}"
        )
    return count


def processVideo(
    inputPath,
    outputDir,
    config=None,
    mode="classical",
    confidence=0.25,
    tracking=True,
    roi=None,
    line=None,
    trails=True,
    progress=None,
    maxFrames=None,
):
    inputPath, outputDir = Path(inputPath), Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)
    for folder in ("videos", "frames", "charts"):
        (outputDir / folder).mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(inputPath))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Cannot read input video. Upload a valid MP4 file.")
    sourceFps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(sourceFps) or sourceFps <= 0:
        capture.release()
        raise ValueError("Video has no valid source FPS")
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(3)), int(capture.get(4))
    filename = (
        "opencv_segmented_output.mp4"
        if mode == "classical"
        else f"yolo_{mode}_output.mp4"
    )
    outputPath = outputDir / "videos" / filename
    if inputPath.resolve() == outputPath.resolve():
        capture.release()
        raise ValueError("Input and output videos must differ")
    writer = None
    rows, objects, savedFrames = [], [], []
    wallStart = time.perf_counter()
    try:
        engine = FrameProcessor(config, mode, confidence, tracking, roi, line, trails)
        writer = cv2.VideoWriter(
            str(outputPath), cv2.VideoWriter_fourcc(*"mp4v"), sourceFps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("MP4 writer unavailable")
        selected = {1, max(1, expected // 2), max(1, expected - 1)}
        while maxFrames is None or len(rows) < maxFrames:
            ok, frame = capture.read()
            if not ok:
                break
            number = len(rows) + 1
            stages, row, detections = engine.process(frame, number, sourceFps)
            writer.write(stages["tracked"])
            rows.append(row)
            for det in detections:
                objects.append(
                    {
                        "frame_number": number,
                        "track_id": det.get("track_id"),
                        "label": det["label"],
                        "center_x": det["center"][0],
                        "center_y": det["center"][1],
                        "bbox": list(det["bbox"]),
                    }
                )
            if number in selected:
                savedFrames.append(number)
                for name, image in stages.items():
                    if not cv2.imwrite(
                        str(outputDir / "frames" / f"frame_{number:04d}_{name}.png"),
                        image,
                    ):
                        raise RuntimeError("Could not write representative frame")
                if number == max(1, expected // 2) and mode == "classical":
                    pipelineFigure(stages, outputDir / "charts" / "pipeline_stages.png")
            if progress and (number % 10 == 0 or number == expected):
                progress(number, expected)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not rows:
        raise ValueError("Video contains no decodable frames")
    if maxFrames is None and expected > 0 and len(rows) != expected:
        raise RuntimeError(
            f"Input ended early: decoded {len(rows)} of {expected} frames"
        )
    codec = browserEncode(outputPath)
    validateVideo(outputPath, len(rows))
    df = pd.DataFrame(rows)
    df.to_csv(outputDir / "frame_metrics.csv", index=False)
    pd.DataFrame(
        objects,
        columns=["frame_number", "track_id", "label", "center_x", "center_y", "bbox"],
    ).to_csv(outputDir / "detections.csv", index=False)
    summary = summarizeMetrics(df, len(engine.seenIds))
    summary.update(
        {
            "mode": mode,
            "source_fps": sourceFps,
            "width": width,
            "height": height,
            "source_frame_count": expected,
            "codec": codec,
            "processing_export_validation_seconds": time.perf_counter() - wallStart,
            "entered": engine.analytics.entered,
            "exited": engine.analytics.exited,
        }
    )
    pd.DataFrame([summary]).to_csv(outputDir / "summary_metrics.csv", index=False)
    createCharts(df, outputDir / "charts")
    manifest = {
        "summary": summary,
        "config": asdict(engine.config),
        "roi": roi,
        "line": line,
        "tracking": tracking,
        "confidence": confidence,
        "representative_frames": savedFrames,
        "video": f"videos/{filename}",
        "input_name": inputPath.name,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "timing_scope": "Preprocess/inference + association + analytics + overlays; excludes decode, HUD, encoding, disk I/O and model load. First inference included.",
    }
    (outputDir / "run.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
