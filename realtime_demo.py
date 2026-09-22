"""Local camera preview; q exits. No browser webcam access is implied."""

import argparse
import cv2
from src.video_processor import FrameProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        print(
            f"Camera {args.camera} unavailable. Check device permissions or use the built-in video demo."
        )
        return 1
    engine = FrameProcessor()
    number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera stopped delivering frames.")
                break
            number += 1
            stages, _, _ = engine.process(
                frame, number, capture.get(cv2.CAP_PROP_FPS) or 30
            )
            cv2.imshow("Vision Intelligence | q to exit", stages["tracked"])
            if cv2.waitKey(1) & 255 == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
