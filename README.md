# Real-time Hand Gesture Blocks (OpenCV + MediaPipe)

Academic, rule-based hand gesture interaction demo:
- Captures webcam frames (OpenCV)
- Tracks hand landmarks (MediaPipe Hands)
- Detects gestures (open palm, pinch) via simple geometric rules
- Maps gestures to actions on on-screen virtual blocks (create, select, move)

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python -m src.app
```

## Controls

- **Open palm (hold ~0.5s)**: create a new block at the palm position
- **Pinch** (thumb tip + index tip together): select nearest block and move it while pinching
- Press `q` or `Esc` to quit

## Notes

This is intentionally rule-based (no ML gesture classifier). Accuracy depends on lighting, camera position, and hand orientation.
