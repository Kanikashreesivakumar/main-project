from __future__ import annotations

import argparse
import time
from dataclasses  import dataclass
from typing import Optional

import cv2
import numpy as np

from .blocks import BlockWorld
from .gestures import GestureDetector, GestureStateMachine
from .hand_tracking import HandTracker


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    mirror: bool = True
    max_num_hands: int = 1
    camera_verbose: bool = False
    camera_backend: str = "AUTO"  # AUTO, DSHOW, MSMF


def _list_cameras(max_index: int = 10) -> int:
    print("[camera] Probing camera indices...")
    found_any = False

    backends: list[tuple[str, int]] = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("DSHOW", int(cv2.CAP_DSHOW)))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(("MSMF", int(cv2.CAP_MSMF)))
    backends.append(("AUTO", 0))

    for idx in range(0, max_index + 1):
        ok_backends: list[str] = []
        for backend_name, backend in backends:
            cap = cv2.VideoCapture(idx, backend) if backend != 0 else cv2.VideoCapture(idx)
            ok = cap.isOpened()
            cap.release()
            if ok:
                ok_backends.append(backend_name)

        if ok_backends:
            found_any = True
            backends_txt = ",".join(ok_backends)
            print(f"[camera] index={idx} OK (backends: {backends_txt})")

    if not found_any:
        print("[camera] No cameras opened via OpenCV.")
        print("[camera] Tips: close Zoom/Teams/Camera app; enable Windows Settings → Privacy & security → Camera → Allow desktop apps.")
        print("[camera] If using Remote Desktop/VM, ensure camera passthrough is enabled.")
        return 1

    return 0


def _parse_args() -> AppConfig | tuple[AppConfig, bool, int]:
    parser = argparse.ArgumentParser(description="Real-time Hand Gesture Blocks (OpenCV + MediaPipe)")
    parser.add_argument("--camera-index", type=int, default=0, help="Preferred webcam index (try 0, 1, 2, …)")
    parser.add_argument(
        "--backend",
        type=str,
        default="AUTO",
        choices=["AUTO", "DSHOW", "MSMF"],
        help="Force camera backend (Windows: DSHOW often works when AUTO fails)",
    )
    parser.add_argument("--max-hands", type=int, default=1, help="Maximum number of hands to track")
    parser.add_argument("--no-mirror", action="store_true", help="Disable mirroring (default is mirrored)")
    parser.add_argument("--camera-verbose", action="store_true", help="Print camera backend/index attempts")
    parser.add_argument("--list-cameras", action="store_true", help="Probe camera indices and exit")
    parser.add_argument("--probe-max-index", type=int, default=10, help="Max index to probe when using --list-cameras")
    ns = parser.parse_args()

    cfg = AppConfig(
        camera_index=ns.camera_index,
        mirror=not ns.no_mirror,
        max_num_hands=max(1, int(ns.max_hands)),
        camera_verbose=bool(ns.camera_verbose),
        camera_backend=str(ns.backend),
    )
    return cfg, bool(ns.list_cameras), int(ns.probe_max_index)


def _open_camera(preferred_index: int, *, verbose: bool = False, backend_preference: str = "AUTO") -> cv2.VideoCapture:
      
    backends: list[tuple[str, int]] = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("DSHOW", int(cv2.CAP_DSHOW)))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(("MSMF", int(cv2.CAP_MSMF)))
    backends.append(("AUTO", 0)) 

    backend_preference = (backend_preference or "AUTO").upper()
    if backend_preference != "AUTO":
        backends = [b for b in backends if b[0] == backend_preference]
        if not backends:
            backends = [("AUTO", 0)]

    indices_to_try = [preferred_index] + [i for i in range(0, 6) if i != preferred_index]

    attempted: list[str] = []

    for backend_name, backend in backends:
        for idx in indices_to_try:
            attempted.append(f"{backend_name}:{idx}")
            cap = cv2.VideoCapture(idx, backend) if backend != 0 else cv2.VideoCapture(idx)
            if cap.isOpened():
                if verbose:
                    print(f"[camera] Opened backend={backend_name} index={idx}")
                return cap
            cap.release()

    if verbose:
        print("[camera] Failed attempts:", ", ".join(attempted))

    raise RuntimeError(
        "Could not open a webcam. Tried indices 0-5 across common backends (preferred index: "
        f"{preferred_index}). Close other apps using the camera and check Windows camera "
        "permissions for desktop apps."
    )


def _draw_landmarks(frame_bgr: np.ndarray, landmarks_px: dict[int, tuple[float, float]]) -> None:
    for idx, (x, y) in landmarks_px.items():
        if idx in (4, 8):
            color = (255, 255, 255)
            r = 6
        else:
            color = (90, 255, 90)
            r = 3
        cv2.circle(frame_bgr, (int(x), int(y)), r, color, -1)


def _draw_hud(frame_bgr: np.ndarray, fps: float, gesture_text: str) -> None:
    lines = [
        f"FPS: {fps:.1f}",
        f"Gesture: {gesture_text}",
        "Open palm (hold): create block",
        "Pinch: select + move",
        "q / Esc: quit",
    ]

    x, y = 12, 28
    for i, t in enumerate(lines):
        yy = y + i * 22
        cv2.putText(frame_bgr, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)


def run(config: AppConfig) -> None:
    cap = _open_camera(
        config.camera_index,
        verbose=config.camera_verbose,
        backend_preference=config.camera_backend,
    )

    tracker = HandTracker(max_num_hands=config.max_num_hands)
    detector = GestureDetector(pinch_threshold=0.35)
    gesture_sm = GestureStateMachine(open_palm_hold_s=0.5, create_cooldown_s=1.0)
    world = BlockWorld()

    prev_t = time.perf_counter()
    fps_ema: Optional[float] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if config.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(now - prev_t, 1e-6)
            prev_t = now
            fps = 1.0 / dt
            fps_ema = fps if fps_ema is None else (0.9 * fps_ema + 0.1 * fps)

            hands = tracker.process(frame)

            gesture_text = "None"
            if hands:
                hand = hands[0]
                snapshot = detector.detect(hand)
                events = gesture_sm.update(snapshot, now_s=time.monotonic())

                if snapshot.pinch and snapshot.pinch_point is not None:
                    gesture_text = "Pinch"
                    if events.pinch_started:
                        world.select_nearest(snapshot.pinch_point, max_dist_px=1.2 * hand.hand_size_px)
                    world.move_selected_to(snapshot.pinch_point)
                elif snapshot.open_palm:
                    gesture_text = "OpenPalm"
                    if events.open_palm_held:
                        world.create_block(hand.palm_center)
                else:
                    if events.pinch_ended:
                        world.clear_selection()

                _draw_landmarks(frame, hand.landmarks_px)

                # Visualize pinch point
                if snapshot.pinch_point is not None:
                    cv2.circle(frame, (int(snapshot.pinch_point[0]), int(snapshot.pinch_point[1])), 10, (255, 255, 255), 2)

            world.draw(frame)
            _draw_hud(frame, fps=fps_ema or fps, gesture_text=gesture_text)

            cv2.imshow("Hand Gesture Blocks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config, list_cameras, probe_max_index = _parse_args()
    if list_cameras:
        raise SystemExit(_list_cameras(max_index=probe_max_index))
    run(config)
   