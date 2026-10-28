"""Finger Lens: your fingertips form a live box; a filter applies only inside it.

Keys:  1-6 switch filter | s toggle hand skeleton | b toggle border | q / ESC quit
"""
import argparse
import time

import cv2

from filters import FILTERS
from lens import LensBox, apply_lens, draw_border
from tracker import HandTracker


def parse_args():
    p = argparse.ArgumentParser(description="Fingertip-controlled filter lens")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--smoothing", type=float, default=0.5,
                   help="0-1, lower = smoother but laggier")
    return p.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    # Many webcams ignore the requested size and fall back to 640x480 (4:3).
    # Show what we actually got, and let the window be resized/maximized.
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {got_w}x{got_h}")
    cv2.namedWindow("Finger Lens", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Finger Lens", 1280, 720)

    tracker = HandTracker(max_hands=2)
    lens = LensBox(smoothing=args.smoothing)
    names = list(FILTERS.keys())
    current = 0
    show_skeleton = True
    show_border = True
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror so it feels natural

        tips = tracker.process(frame)
        lens.update(tips)

        state = lens.state()
        if state is not None:
            pts, opacity = state
            frame = apply_lens(frame, pts, FILTERS[names[current]], opacity)
            if show_border:
                draw_border(frame, pts)

        if show_skeleton:
            tracker.draw_skeleton(frame)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        mode = lens.mode if state is not None else "show your hands"
        cv2.putText(frame, f"[{current + 1}] {names[current]}  |  {mode}  |  {fps:.0f} FPS",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Finger Lens", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            show_skeleton = not show_skeleton
        if key == ord("b"):
            show_border = not show_border
        if ord("1") <= key < ord("1") + len(names):
            current = key - ord("1")

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
