"""Filters applied inside the lens. Each takes a BGR image and returns a BGR image
of the same size."""
import cv2
import numpy as np


def invert(img):
    return cv2.bitwise_not(img)


def grayscale(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def blur(img):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=12)


def pixelate(img, block=14):
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def edges(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    e = cv2.Canny(g, 60, 150)
    return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)


def thermal(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(g, cv2.COLORMAP_JET)


# Order matters: keys 1..N on the keyboard map to this order.
FILTERS = {
    "Invert": invert,
    "Grayscale": grayscale,
    "Blur": blur,
    "Pixelate": pixelate,
    "Edges": edges,
    "Thermal": thermal,
}
