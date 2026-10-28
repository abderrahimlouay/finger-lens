"""Lens geometry: turns fingertips into a smoothed quadrilateral and composites
a filter inside it."""
import cv2
import numpy as np


def order_points(pts):
    """Order 4 points clockwise starting from the top-left-most one, so the
    polygon never self-intersects and point identity stays stable over time."""
    pts = np.asarray(pts, dtype=np.float64)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(angles)]  # clockwise on screen (y points down)
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def rect_from_two(a, b):
    """Axis-aligned rectangle with a and b as opposite corners."""
    x0, x1 = sorted([a[0], b[0]])
    y0, y1 = sorted([a[1], b[1]])
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


class LensBox:
    """Holds the current box, smooths it, and keeps it alive briefly when
    tracking drops so it fades out instead of flickering."""

    def __init__(self, smoothing=0.5, hold_frames=6, fade_frames=8):
        self.alpha = smoothing  # 1.0 = no smoothing, lower = smoother
        self.hold = hold_frames
        self.fade = fade_frames
        self.pts = None
        self.missing = 0
        self.mode = None  # "two-hand" / "one-hand"

    def update(self, tips):
        """tips: list of (thumb, index) pairs, one per detected hand."""
        new = None
        if len(tips) >= 2:
            raw = [tips[0][0], tips[0][1], tips[1][0], tips[1][1]]
            new = order_points(raw)
            self.mode = "two-hand"
        elif len(tips) == 1:
            new = rect_from_two(tips[0][0], tips[0][1])
            self.mode = "one-hand"

        if new is not None:
            if self.pts is None:
                self.pts = new
            else:
                self.pts = self.alpha * new + (1 - self.alpha) * self.pts
            self.missing = 0
        else:
            self.missing += 1

    def state(self):
        """Returns (points, opacity) or None if there is nothing to draw."""
        if self.pts is None:
            return None
        if self.missing <= self.hold:
            return self.pts, 1.0
        t = (self.missing - self.hold) / max(1, self.fade)
        if t >= 1.0:
            self.pts = None
            return None
        return self.pts, 1.0 - t


def apply_lens(frame, pts, filter_fn, opacity=1.0):
    """Apply filter_fn only inside polygon pts. Works on the polygon's bounding
    box for speed, then masks to the exact shape."""
    h, w = frame.shape[:2]
    poly = np.round(pts).astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(poly)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + bw, w), min(y + bh, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return frame

    roi = frame[y0:y1, x0:x1]
    filtered = filter_fn(roi)
    if opacity < 1.0:
        filtered = cv2.addWeighted(filtered, opacity, roi, 1.0 - opacity, 0)

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly - np.array([x0, y0], dtype=np.int32)], 255)

    out = frame.copy()
    out[y0:y1, x0:x1] = np.where(mask[..., None] > 0, filtered, roi)
    return out


def draw_border(frame, pts, color=(0, 255, 255), thickness=2):
    poly = np.round(pts).astype(np.int32)
    cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=thickness)
    for p in poly:
        cv2.circle(frame, tuple(int(v) for v in p), 6, color, -1)
