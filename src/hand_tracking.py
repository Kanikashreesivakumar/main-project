from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

Point2 = Tuple[float, float]


@dataclass(frozen=True)
class HandObservation:
    handedness: str  # "Left" or "Right" (MediaPipe's label)
    landmarks_px: Dict[int, Point2]  # landmark index -> (x, y) in pixels
    landmarks_norm: Dict[int, Tuple[float, float, float]]  # landmark index -> (x, y, z) normalized
    hand_size_px: float  # scale proxy for thresholds

    @property
    def wrist(self) -> Point2:
        return self.landmarks_px[0]

    @property
    def palm_center(self) -> Point2:
        # A stable-ish palm reference: midpoint of wrist (0) and middle MCP (9)
        w = np.array(self.landmarks_px[0], dtype=np.float32)
        m = np.array(self.landmarks_px[9], dtype=np.float32)
        c = (w + m) / 2.0
        return float(c[0]), float(c[1])


class HandTracker:
    """MediaPipe Hands wrapper.

    Responsibilities:
    - Run landmark inference on frames
    - Convert landmarks to pixel coordinates
    - Provide a hand scale estimate (hand_size_px)

    This module intentionally does NOT interpret gestures.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
        model_complexity: int = 0,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def process(self, frame_bgr: np.ndarray) -> List[HandObservation]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            return []

        observations: List[HandObservation] = []

        handedness_labels: List[str] = []
        if result.multi_handedness:
            for h in result.multi_handedness:
                handedness_labels.append(h.classification[0].label)

        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            handedness = handedness_labels[i] if i < len(handedness_labels) else "Unknown"

            landmarks_px: Dict[int, Point2] = {}
            landmarks_norm: Dict[int, Tuple[float, float, float]] = {}

            for idx, lm in enumerate(hand_landmarks.landmark):
                x_px = float(lm.x * width)
                y_px = float(lm.y * height)
                landmarks_px[idx] = (x_px, y_px)
                landmarks_norm[idx] = (float(lm.x), float(lm.y), float(lm.z))

            # Scale proxy: wrist (0) to middle MCP (9)
            w = np.array(landmarks_px[0], dtype=np.float32)
            m = np.array(landmarks_px[9], dtype=np.float32)
            hand_size_px = float(np.linalg.norm(m - w))
            hand_size_px = max(hand_size_px, 1.0)

            observations.append(
                HandObservation(
                    handedness=handedness,
                    landmarks_px=landmarks_px,
                    landmarks_norm=landmarks_norm,
                    hand_size_px=hand_size_px,
                )
            )

        return observations
