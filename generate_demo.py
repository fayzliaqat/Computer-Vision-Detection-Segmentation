"""Deterministic, non-overlapping moving shapes and frame-level ground truth."""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.video_processor import browserEncode, validateVideo

ROOT = Path(__file__).resolve().parent


def generateDemo(output=None, frames=240, fps=30, seed=42):
    output = Path(output or ROOT / "data" / "sample_shapes.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    width, height = 960, 540
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot create synthetic video")
    truth = []
    try:
        for index in range(frames):
            gradient = np.linspace(228, 244, width)[None, :] + 2 * np.sin(index / 30)
            gray = np.clip(
                np.broadcast_to(gradient, (height, width))
                + rng.normal(0, 1, (height, width)),
                0,
                255,
            ).astype(np.uint8)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Separate lanes make correspondence unambiguous; this is an easy benchmark.
            objects = [
                (1, "Circle", 80 + int(index * 2.9), 105, 29),
                (2, "Rectangle", 850 - int(index * 2.8), 230, 32),
                (3, "Triangle", 100 + int(index * 2.6), 365, 35),
            ]
            if 45 <= index < 190:
                objects.append((4, "Circle", 770 - int((index - 45) * 2.2), 455, 19))
            for objectId, shape, x, y, size in objects:
                color = (45, 65, 80)
                if shape == "Circle":
                    cv2.circle(frame, (x, y), size, color, -1)
                elif shape == "Rectangle":
                    cv2.rectangle(
                        frame, (x - size, y - 23), (x + size, y + 23), color, -1
                    )
                else:
                    vertices = np.array(
                        [
                            [x, y - size],
                            [x - size, y + size // 2],
                            [x + size, y + size // 2],
                        ],
                        np.int32,
                    )
                    cv2.fillPoly(frame, [vertices], color)
                truth.append(
                    {
                        "frame_number": index + 1,
                        "object_id": objectId,
                        "shape": shape,
                        "center_x": x,
                        "center_y": y,
                    }
                )
            writer.write(frame)
    finally:
        writer.release()
    browserEncode(output)
    validateVideo(output, frames)
    pd.DataFrame(truth).to_csv(output.with_name("ground_truth.csv"), index=False)
    return output


if __name__ == "__main__":
    print(generateDemo())
