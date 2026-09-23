"""Small greedy one-to-one centroid tracker, implemented from scratch."""

from collections import deque
import math


class CentroidTracker:
    def __init__(self, maxDistance=55, maxDisappeared=10):
        self.maxDistance = maxDistance
        self.maxDisappeared = maxDisappeared
        self.nextId = 1
        self.tracks = {}

    def update(self, detections):
        candidates = sorted(
            (math.dist(track["center"], det["center"]), trackId, index)
            for trackId, track in self.tracks.items()
            for index, det in enumerate(detections)
        )
        usedTracks, usedDetections = set(), set()
        for distance, trackId, index in candidates:
            if distance > self.maxDistance:
                break
            if trackId in usedTracks or index in usedDetections:
                continue
            self.tracks[trackId]["center"] = detections[index]["center"]
            self.tracks[trackId]["missed"] = 0
            self.tracks[trackId]["history"].append(detections[index]["center"])
            detections[index]["track_id"] = trackId
            usedTracks.add(trackId)
            usedDetections.add(index)
        for trackId in list(self.tracks):
            if trackId not in usedTracks:
                self.tracks[trackId]["missed"] += 1
                if self.tracks[trackId]["missed"] > self.maxDisappeared:
                    del self.tracks[trackId]
        for index, det in enumerate(detections):
            if index not in usedDetections:
                trackId = self.nextId
                self.nextId += 1
                self.tracks[trackId] = {
                    "center": det["center"],
                    "missed": 0,
                    "history": deque([det["center"]], maxlen=24),
                }
                det["track_id"] = trackId
        return detections


class SceneAnalytics:
    """ROI uses centroid inclusion; one crossing per ID per run, with deadband."""

    def __init__(
        self,
        roi=None,
        line=None,
        deadband=4,
        lineOrientation="vertical",
        trailLength=24,
    ):
        if roi is not None and not (
            0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1
        ):
            raise ValueError("ROI must be normalized left, top, right, bottom")
        if line is not None and not 0 < line < 1:
            raise ValueError("Line position must be between zero and one")
        if lineOrientation not in ("vertical", "horizontal"):
            raise ValueError("Line orientation must be vertical or horizontal")
        if not 2 <= trailLength <= 60:
            raise ValueError("Trail length must be between 2 and 60 frames")
        self.lineOrientation, self.trailLength = lineOrientation, trailLength
        self.events = []
        self.roi, self.line, self.deadband = roi, line, deadband
        self.sides, self.counted, self.histories, self.lastSeen = {}, set(), {}, {}
        self.entered = self.exited = self.frame = 0

    def update(self, detections, width, height):
        self.frame += 1
        roiCounts = {}
        for det in detections:
            x, y = det["center"]
            inside = self.roi is None or (
                self.roi[0] * width <= x <= self.roi[2] * width
                and self.roi[1] * height <= y <= self.roi[3] * height
            )
            det["in_roi"] = inside
            if inside:
                roiCounts[det["label"]] = roiCounts.get(det["label"], 0) + 1
            trackId = det.get("track_id")
            if trackId is None:
                continue
            self.histories.setdefault(trackId, deque(maxlen=self.trailLength)).append(
                (x, y)
            )
            self.lastSeen[trackId] = self.frame
            if self.line is not None:
                coordinate, extent = (
                    (x, width) if self.lineOrientation == "vertical" else (y, height)
                )
                delta = coordinate - self.line * extent
                side = (
                    1 if delta > self.deadband else -1 if delta < -self.deadband else 0
                )
                prior = self.sides.get(trackId)
                if side and prior and side != prior and trackId not in self.counted:
                    self.entered += int(side == 1)
                    self.exited += int(side == -1)
                    self.counted.add(trackId)
                    self.events.append(
                        {
                            "frame_number": self.frame,
                            "track_id": trackId,
                            "class": det["label"],
                            "direction": "A" if side == 1 else "B",
                            "center_x": x,
                            "center_y": y,
                        }
                    )
                if side:
                    self.sides[trackId] = side
        for trackId in list(self.lastSeen):
            if self.frame - self.lastSeen[trackId] > 30:
                self.histories.pop(trackId, None)
                self.sides.pop(trackId, None)
                del self.lastSeen[trackId]
        return roiCounts
