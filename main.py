"""
Sentiment Analysis — real-time multi-person facial emotion overlay.

Run:
    python main.py                     # default mode from config (both)
    python main.py --mode emoji
    python main.py --mode text
    python main.py --camera 1

Live keys:
    E = emoji only      T = text only      B = both
    Q / Esc = quit
"""

import argparse
import time

import cv2
import config
from emotion_engine import EmotionEngine
from tracker import FaceTracker
from overlay import OverlayRenderer

WINDOW = "Sentiment Analysis"


def parse_args():
    p = argparse.ArgumentParser(description="Real-time facial emotion overlay")
    p.add_argument("--mode", choices=["emoji", "text", "both"],
                   default=config.DISPLAY_MODE)
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    return p.parse_args()


def draw_hud(frame, fps, mode, n_faces, device):
    h = frame.shape[0]
    hud = (f"FPS {fps:4.1f}   mode: {mode}   faces: {n_faces}   "
           f"{device}   [E]moji [T]ext [B]oth [Q]uit")
    cv2.putText(frame, hud, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, hud, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (240, 240, 240), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    mode = args.mode

    print("Loading models (YuNet + AffectNet EfficientNet, ONNX)...")
    engine = EmotionEngine()
    tracker = FaceTracker()
    renderer = OverlayRenderer()
    print("Emoji source:",
          "system color-emoji font" if renderer.using_font else "built-in drawn fallback")
    print("Inference device:", engine.provider_label)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    frame_i = 0
    fps, t_prev = 0.0, time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed; exiting.")
                break
            frame = cv2.flip(frame, 1)  # mirror view feels natural

            # Schedule inference every Nth frame (runs in background thread)
            if frame_i % config.DETECT_EVERY_N == 0:
                engine.submit(frame)
            frame_i += 1

            results, fresh = engine.poll()
            if fresh:
                tracker.update(results)

            renderer.draw(frame, tracker.active, mode)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            draw_hud(frame, fps, mode, len(tracker.active), engine.provider_label)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("e"):
                mode = "emoji"
            elif key == ord("t"):
                mode = "text"
            elif key == ord("b"):
                mode = "both"
    finally:
        engine.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
