"""Geometric contour classification; these labels are not semantic classes."""

import math
import cv2


def classifyShape(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return "Other"
    vertices = len(cv2.approxPolyDP(contour, 0.025 * perimeter, True))
    if vertices == 3:
        return "Triangle"
    if vertices == 4:
        return "Rectangle"
    circularity = 4 * math.pi * area / perimeter**2
    return "Circle" if vertices >= 6 and circularity >= 0.76 else "Other"


def extractContours(mask, config):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not config.minArea <= area <= config.maxArea:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        detections.append(
            {
                "label": classifyShape(contour),
                "center": center,
                "bbox": cv2.boundingRect(contour),
                "area": area,
                "perimeter": cv2.arcLength(contour, True),
                "contour": contour,
            }
        )
    return detections, len(contours)
