# Finger Lens

A webcam app where your fingertips form a live box, and a filter is applied **only inside it**.

- **Two hands:** thumb + index tips of both hands = 4 corners (works tilted, it's a quadrilateral, not just a rectangle).
- **One hand:** thumb tip and index tip = opposite corners of a rectangle.

## Setup

Python 3.9-3.12 recommended (MediaPipe wheels lag behind the newest Python).

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Options: `--camera 1`, `--width 1280 --height 720`, `--smoothing 0.3`.

## Controls

| Key | Action |
|-----|--------|
| 1-6 | Invert, Grayscale, Blur, Pixelate, Edges, Thermal |
| s   | Toggle hand skeleton |
| b   | Toggle box border |
| q / ESC | Quit |

## How it works

1. OpenCV captures and mirrors the frame.
2. MediaPipe Hands returns landmarks; thumb tip = 4, index tip = 8.
3. `lens.py` orders the 4 points clockwise (no self-intersection), smooths them with an exponential moving average, and keeps the box alive for a few frames if tracking drops so it fades instead of flickering.
4. The filter runs on the polygon's bounding box, then a polygon mask composites it back over the original frame.

## Structure

```
main.py      # capture loop, keys, HUD
tracker.py   # MediaPipe wrapper
lens.py      # geometry, smoothing, masking/compositing
filters.py   # the 6 filters (add yours to FILTERS)
```

## Ideas for v2

- Gesture-based filter switching (pinch / fist / open palm)
- Magnify / fisheye inside the box
- Perspective-warp another video or image into the box (`cv2.getPerspectiveTransform`)
