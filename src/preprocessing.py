"""Configurable grayscale, Gaussian, adaptive threshold and morphology stages."""

from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass
class VisionConfig:
    gaussianKernel: int = 5
    blockSize: int = 31
    thresholdC: float = 7
    adaptiveMethod: str = "Gaussian"
    morphologyKernel: int = 3
    minArea: float = 250
    maxArea: float = 20000
    maxDistance: float = 55
    maxDisappeared: int = 10

    def validate(self):
        for name in ("gaussianKernel", "blockSize"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 3 or value % 2 == 0:
                raise ValueError(f"{name} must be an odd integer >= 3")
        if self.morphologyKernel not in (0, 3, 5, 7):
            raise ValueError("Morphology kernel must be 0, 3, 5 or 7")
        if not 0 <= self.minArea < self.maxArea:
            raise ValueError("Area bounds must satisfy 0 <= minimum < maximum")
        if self.adaptiveMethod not in ("Gaussian", "Mean"):
            raise ValueError("Adaptive method must be Gaussian or Mean")
        if self.maxDistance <= 0 or self.maxDisappeared < 0:
            raise ValueError("Invalid tracking settings")
        if not all(
            np.isfinite(v) for v in asdict(self).values() if isinstance(v, (int, float))
        ):
            raise ValueError("Configuration values must be finite")
        return self


def preprocessFrame(frame, config):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (config.gaussianKernel, config.gaussianKernel), 0)
    method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if config.adaptiveMethod == "Gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    threshold = cv2.adaptiveThreshold(
        blurred, 255, method, cv2.THRESH_BINARY_INV, config.blockSize, config.thresholdC
    )
    mask = threshold.copy()
    if config.morphologyKernel:
        kernel = np.ones((config.morphologyKernel, config.morphologyKernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return {
        "original": frame.copy(),
        "grayscale": gray,
        "blurred": blurred,
        "threshold": threshold,
        "mask": mask,
    }
