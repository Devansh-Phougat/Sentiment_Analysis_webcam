"""
Face detection via YuNet (cv2.FaceDetectorYN).

Why YuNet replaces the old Haar cascade:
- It's a small CNN with built-in non-maximum suppression, so ONE face gets
  ONE box (the old Haar detector fired several overlapping boxes on close
  faces - the "multiple rectangles on my face" bug).
- Far better recall/precision on angled, partially lit, and close-up faces.
- Ships inside opencv-python >= 4.8; runs in ~2-5 ms at 640 px on CPU.

Detection runs on a downscaled copy for speed, but boxes are scaled back to
FULL-resolution coordinates so emotion crops keep every pixel of detail.
"""

import cv2
import config


class FaceDetector:
    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(
            config.YUNET_MODEL, "",
            (320, 320),                      # placeholder; set per frame size
            config.FACE_SCORE_THRESHOLD,
            config.NMS_THRESHOLD,
            5000,
        )
        self._size = None

    def detect(self, frame):
        """Return list of ([x, y, w, h], score) in FULL-frame coordinates."""
        fh, fw = frame.shape[:2]
        scale = fw / config.DETECT_WIDTH
        sw, sh = config.DETECT_WIDTH, max(1, int(round(fh / scale)))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        if self._size != (sw, sh):
            self.det.setInputSize((sw, sh))
            self._size = (sw, sh)

        _, faces = self.det.detect(small)
        boxes = []
        if faces is not None:
            for f in faces:
                x, y, w, h = (float(v) * scale for v in f[:4])
                if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
                    continue
                # clamp to frame
                x, y = max(0.0, x), max(0.0, y)
                w = min(w, fw - x)
                h = min(h, fh - y)
                if w <= 0 or h <= 0:
                    continue
                boxes.append(([int(x), int(y), int(w), int(h)], float(f[14])))
        return boxes
