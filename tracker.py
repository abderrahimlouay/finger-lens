"""Hand tracking: wraps MediaPipe Hands and returns thumb/index fingertips."""
import cv2
import numpy as np
import mediapipe as mp

THUMB_TIP = 4
INDEX_TIP = 8


class HandTracker:
    def __init__(self, max_hands=2, detection_conf=0.6, tracking_conf=0.6):
        self._mp_hands = mp.solutions.hands
        self._drawer = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._last = None

    def process(self, frame_bgr):
        """Run detection on a BGR frame. Returns a list of (thumb_tip, index_tip)
        per detected hand, each as a float np.array([x, y]) in pixel coords."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self._last = self._hands.process(rgb)

        tips = []
        if self._last.multi_hand_landmarks:
            for hand in self._last.multi_hand_landmarks:
                t = hand.landmark[THUMB_TIP]
                i = hand.landmark[INDEX_TIP]
                tips.append((
                    np.array([t.x * w, t.y * h], dtype=np.float64),
                    np.array([i.x * w, i.y * h], dtype=np.float64),
                ))
        return tips

    def draw_skeleton(self, frame_bgr):
        if self._last and self._last.multi_hand_landmarks:
            for hand in self._last.multi_hand_landmarks:
                self._drawer.draw_landmarks(
                    frame_bgr, hand, self._mp_hands.HAND_CONNECTIONS
                )

    def close(self):
        self._hands.close()
