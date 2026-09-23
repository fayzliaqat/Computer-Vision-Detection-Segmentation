"""Optional lazy-loaded YOLOv8 detection / instance segmentation + ByteTrack."""

from pathlib import Path
import os
import time


class YoloDetector:
    def __init__(
        self, segmentation=False, confidence=0.25, tracking=True, classNames=None
    ):
        if not 0 < confidence <= 1:
            raise ValueError("Confidence must be in (0, 1]")
        root = Path(__file__).resolve().parents[1]
        (root / "models").mkdir(exist_ok=True)
        configDir = root / "tmp" / "ultralytics"
        configDir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(configDir))
        try:
            from ultralytics import YOLO

            modelName = "yolov8n-seg.pt" if segmentation else "yolov8n.pt"
            self.model = YOLO(str(root / "models" / modelName))
        except Exception as exc:
            raise RuntimeError(
                "YOLO unavailable. Install requirements-yolo.txt and allow the initial model download; classical vision remains available."
            ) from exc
        self.confidence, self.tracking = confidence, tracking
        self.segmentation = segmentation
        self.modelName = modelName
        self.classIds = None
        if classNames is not None:
            self.classIds = [
                index for index, name in self.model.names.items() if name in classNames
            ]
            if not self.classIds or len(self.classIds) != len(set(classNames)):
                raise ValueError("Select at least one supported model class")

    def detect(self, frame):
        start = time.perf_counter()
        args = dict(conf=self.confidence, verbose=False, device="cpu", imgsz=640)
        if self.classIds is not None:
            args["classes"] = self.classIds
        if self.tracking:
            result = self.model.track(
                frame, persist=True, tracker="bytetrack.yaml", **args
            )[0]
        else:
            result = self.model.predict(frame, **args)[0]
        inferenceMs = (time.perf_counter() - start) * 1000
        detections = []
        polygons = result.masks.xy if result.masks is not None else []
        for index, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det = {
                "label": result.names[int(box.cls.item())],
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "confidence": float(box.conf.item()),
                "track_id": int(box.id.item()) if box.id is not None else None,
            }
            if index < len(polygons):
                det["contour"] = polygons[index].astype("int32").reshape(-1, 1, 2)
            detections.append(det)
        return detections, inferenceMs
