"""Vehicle-only scene summaries and readable overlays; no density/accuracy estimates."""

import json
import cv2
import numpy as np
import pandas as pd

VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle")
VEHICLE_COLORS = {
    "car": (186, 190, 94),
    "truck": (118, 181, 220),
    "bus": (204, 155, 124),
    "motorcycle": (170, 142, 198),
    "bicycle": (150, 189, 168),
}
TRAFFIC_FOOTER = 82


def vehicleCounts(detections):
    return {
        label: sum(d["label"] == label for d in detections) for label in VEHICLE_CLASSES
    }


def summarizeTraffic(df):
    counts = pd.DataFrame([json.loads(value) for value in df.vehicle_class_counts])
    return {
        "mean_active_vehicles": float(df.active_vehicles.mean()),
        "peak_active_vehicles": int(df.active_vehicles.max()),
        "final_active_vehicles": int(df.active_vehicles.iloc[-1]),
        "direction_a": int(df.direction_a.iloc[-1]),
        "direction_b": int(df.direction_b.iloc[-1]),
        "total_crossings": int(df.total_crossings.iloc[-1]),
        "mean_inference_time_ms": float(df.inference_time_ms.mean()),
        "vehicle_classes": {
            label: {
                "mean_visible": float(counts[label].mean()),
                "peak_visible": int(counts[label].max()),
            }
            for label in VEHICLE_CLASSES
        },
    }


def trafficOverlay(frame, detections, analytics, trails=True):
    """Draw small distant boxes without crowded text, short trails and thin line."""
    output = frame.copy()
    height, width = frame.shape[:2]
    occupied = []
    # Large foreground targets get readable labels before distant targets.
    for det in sorted(
        detections, key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True
    ):
        x, y, w, h = det["bbox"]
        color = VEHICLE_COLORS[det["label"]]
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
        trackId = det.get("track_id")
        if trails and trackId in analytics.histories:
            points = np.array(analytics.histories[trackId], np.int32)
            if len(points) > 1:
                cv2.polylines(output, [points], False, color, 1, cv2.LINE_AA)
        if w < 40 or h < 26:
            continue
        label = det["label"] + (f" #{trackId}" if trackId is not None else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        lx = min(max(x, 2), max(2, width - tw - 8))
        for ty in (y - 5, y + h + th + 5, y - th - 12):
            box = (lx, ty - th - 4, lx + tw + 6, ty + 4)
            if box[1] < 0 or box[3] >= height:
                continue
            if any(
                box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1]
                for b in occupied
            ):
                continue
            cv2.rectangle(output, box[:2], box[2:], (25, 31, 40), -1)
            cv2.putText(
                output,
                label,
                (lx + 3, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )
            occupied.append(box)
            break
    if analytics.line is not None:
        vertical = analytics.lineOrientation == "vertical"
        pos = int(analytics.line * (width if vertical else height))
        extent = height if vertical else width
        for start in range(0, extent, 24):
            a, b = (pos, start), (pos, min(start + 13, extent - 1))
            if not vertical:
                a, b = a[::-1], b[::-1]
            cv2.line(output, a, b, (215, 214, 200), 1, cv2.LINE_AA)
    return output


def trafficHud(frame, row):
    # Added beneath the image: vehicles are never hidden by the telemetry panel.
    output = cv2.copyMakeBorder(
        frame, 0, TRAFFIC_FOOTER, 0, 0, cv2.BORDER_CONSTANT, value=(30, 23, 15)
    )
    height, width = frame.shape[:2]
    lines = [
        f"TRAFFIC FLOW  |  FRAME {row['frame_number']}  |  ACTIVE VEHICLES {row['active_vehicles']}  |  CROSSINGS {row['total_crossings']}  [A {row['direction_a']} / B {row['direction_b']}]",
        f"{row['processing_time_ms']:.1f} ms processing  |  {row['fps']:.1f} processing FPS  |  "
        + "   ".join(
            f"{k} {v}" for k, v in json.loads(row["vehicle_class_counts"]).items() if v
        ),
        "A: top to bottom  |  B: bottom to top"
        if row["line_orientation"] == "horizontal"
        else "A: left to right  |  B: right to left",
    ]
    for index, text in enumerate(lines):
        scale = min(
            0.55,
            (width - 30)
            / max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0], 1),
        )
        cv2.putText(
            output,
            text,
            (14, height + 22 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (227, 233, 239),
            1,
            cv2.LINE_AA,
        )
    return output
