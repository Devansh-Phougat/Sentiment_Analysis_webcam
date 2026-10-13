"""
Lightweight multi-face tracker (v2).

Detection+emotion runs every Nth frame; tracks keep overlays glued to each
person in between.

What changed vs v1: instead of debouncing hard LABELS by vote counting, each
track now smooths the full 7-dim PROBABILITY vector with an EMA and only
switches its displayed emotion when the challenger beats the incumbent by a
margin (hysteresis). This uses everything the model says each frame - a
60/40 happy-vs-neutral face converges to 'happy' quickly and stably instead
of flickering or getting stuck.
"""

import itertools

import numpy as np

import config

_EMO_INDEX = {e: i for i, e in enumerate(config.EMOTIONS)}


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class Track:
    _ids = itertools.count(1)

    def __init__(self, box, probs):
        self.id = next(Track._ids)
        self.box = [float(v) for v in box]
        self.probs = np.asarray(probs, dtype=np.float32).copy()
        self.emotion = config.EMOTIONS[int(self.probs.argmax())]
        self.misses = 0

    @property
    def score(self):
        return float(self.probs[_EMO_INDEX[self.emotion]])

    def update(self, box, probs):
        a = config.BOX_SMOOTHING
        for i in range(4):
            self.box[i] = a * box[i] + (1 - a) * self.box[i]
        self.misses = 0

        ap = config.PROB_SMOOTHING
        self.probs = ap * np.asarray(probs, dtype=np.float32) + (1 - ap) * self.probs

        # hysteresis: challenger must clearly beat the incumbent
        best = int(self.probs.argmax())
        cur = _EMO_INDEX[self.emotion]
        if best != cur and self.probs[best] > self.probs[cur] + config.SWITCH_MARGIN:
            self.emotion = config.EMOTIONS[best]


class FaceTracker:
    def __init__(self):
        self.tracks = []

    def update(self, detections):
        """detections: list of (box, probs7). Call once per inference cycle."""
        unmatched = list(detections)
        for track in self.tracks:
            best, best_iou = None, config.IOU_MATCH_THRESHOLD
            for det in unmatched:
                iou = _iou(track.box, det[0])
                if iou > best_iou:
                    best, best_iou = det, iou
            if best is not None:
                track.update(*best)
                unmatched.remove(best)
            else:
                track.misses += 1

        for det in unmatched:
            self.tracks.append(Track(*det))

        self.tracks = [t for t in self.tracks if t.misses <= config.TRACK_MAX_MISSES]

    @property
    def active(self):
        return self.tracks
