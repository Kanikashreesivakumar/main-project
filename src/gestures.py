from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .hand_tracking import HandObservation, Point2


def _dist(a: Point2, b: Point2) -> float:
    ax, ay = a
    bx, by = b
    return float(np.hypot(ax - bx, ay - by))


def _finger_extended(landmarks: Dict[int, Point2], tip: int, pip: int, min_delta_px: float = 8.0) -> bool:
    # Image coordinate system: y increases downward.
    # Finger considered extended if tip is above (smaller y) its PIP joint by a margin.
    tip_y = landmarks[tip][1]
    pip_y = landmarks[pip][1]
    return (pip_y - tip_y) > min_delta_px


def _thumb_extended(landmarks: Dict[int, Point2], handedness: str) -> bool:
    # Simple heuristic: thumb tip is "outside" relative to the hand.
    # For a mirrored display (common in webcam UIs), handedness labels remain MediaPipe's.
    # We use x-axis separation between thumb tip (4) and index MCP (5).
    thumb_tip_x = landmarks[4][0]
    index_mcp_x = landmarks[5][0]

    if handedness == "Right":
        return thumb_tip_x < index_mcp_x - 10.0
    if handedness == "Left":
        return thumb_tip_x > index_mcp_x + 10.0
    return abs(thumb_tip_x - index_mcp_x) > 15.0


@dataclass(frozen=True)
class GestureSnapshot:
    open_palm: bool
    pinch: bool
    pinch_point: Optional[Point2]


class GestureDetector:
    """Rule-based gesture detector.

    Produces per-frame gesture booleans and a pinch point when applicable.
    """

    def __init__(self, pinch_threshold: float = 0.35) -> None:
        # pinch_threshold is relative to hand_size_px
        self._pinch_threshold = pinch_threshold

    def detect(self, hand: HandObservation) -> GestureSnapshot:
        lm = hand.landmarks_px

        pinch_dist = _dist(lm[4], lm[8])
        pinch = pinch_dist < (self._pinch_threshold * hand.hand_size_px)

        # Open palm: index, middle, ring, pinky extended + thumb extended.
        # This is deliberately simple and may fail for rotated hands.
        index_ext = _finger_extended(lm, tip=8, pip=6)
        middle_ext = _finger_extended(lm, tip=12, pip=10)
        ring_ext = _finger_extended(lm, tip=16, pip=14)
        pinky_ext = _finger_extended(lm, tip=20, pip=18)
        thumb_ext = _thumb_extended(lm, hand.handedness)

        open_palm = all([index_ext, middle_ext, ring_ext, pinky_ext, thumb_ext]) and not pinch

        pinch_point: Optional[Point2] = None
        if pinch:
            # midpoint between thumb tip and index tip
            a = np.array(lm[4], dtype=np.float32)
            b = np.array(lm[8], dtype=np.float32)
            p = (a + b) / 2.0
            pinch_point = float(p[0]), float(p[1])

        return GestureSnapshot(open_palm=open_palm, pinch=pinch, pinch_point=pinch_point)


@dataclass
class GestureEvents:
    open_palm_held: bool
    pinch_started: bool
    pinch_ended: bool


class GestureStateMachine:
    """Temporal logic: converts per-frame snapshots into debounced events."""

    def __init__(self, open_palm_hold_s: float = 0.5, create_cooldown_s: float = 1.0) -> None:
        self._open_palm_hold_s = open_palm_hold_s
        self._create_cooldown_s = create_cooldown_s

        self._open_palm_since: Optional[float] = None
        self._last_create_time: float = -1e9

        self._pinch_prev: bool = False

    def update(self, snapshot: GestureSnapshot, now_s: float) -> GestureEvents:
        pinch_started = snapshot.pinch and not self._pinch_prev
        pinch_ended = (not snapshot.pinch) and self._pinch_prev
        self._pinch_prev = snapshot.pinch

        open_palm_held = False
        if snapshot.open_palm:
            if self._open_palm_since is None:
                self._open_palm_since = now_s
            held_for = now_s - self._open_palm_since
            if held_for >= self._open_palm_hold_s and (now_s - self._last_create_time) >= self._create_cooldown_s:
                open_palm_held = True
                self._last_create_time = now_s
                # reset so the user must "re-hold" for another create
                self._open_palm_since = now_s
        else:
            self._open_palm_since = None

        return GestureEvents(open_palm_held=open_palm_held, pinch_started=pinch_started, pinch_ended=pinch_ended)
