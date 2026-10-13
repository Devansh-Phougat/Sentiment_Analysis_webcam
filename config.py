"""
Sentiment Analysis - configuration (v2, ONNX stack)
All tunable knobs live here.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# ----- The 7 emotions (sarcasm included) -----
EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "neutral", "sarcasm"]

EMOJI = {
    "happy":    "😄",
    "sad":      "😢",
    "angry":    "😠",
    "surprise": "😲",
    "fear":     "😨",
    "neutral":  "😐",
    "sarcasm":  "😏",
}

# BGR colors for the text pill / accents (one per emotion)
COLORS = {
    "happy":    (60, 200, 60),
    "sad":      (220, 130, 40),
    "angry":    (50, 50, 230),
    "surprise": (0, 190, 255),
    "fear":     (200, 60, 200),
    "neutral":  (160, 160, 160),
    "sarcasm":  (0, 220, 220),
}

# ----- Display -----
DISPLAY_MODE = "both"        # "emoji" | "text" | "both"  (toggle live with E/T/B keys)
EMOJI_BASE_SIZE = 96
BOB_AMPLITUDE = 6
BOB_SPEED = 2.2

# ----- Camera -----
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ----- Face detection (YuNet via cv2.FaceDetectorYN) -----
YUNET_MODEL = os.path.join(_HERE, "models", "yunet_n_640_640.onnx")
DETECT_WIDTH = 640           # detection runs on a 640px-wide copy (crops come from full res!)
DETECT_EVERY_N = 2           # run detection+emotion every Nth frame
FACE_SCORE_THRESHOLD = 0.65  # YuNet confidence gate
NMS_THRESHOLD = 0.35         # YuNet built-in NMS (kills duplicate boxes on one face)
MIN_FACE_SIZE = 48           # ignore faces smaller than this (px, full-res frame)

# ----- Emotion model (HSEmotion EfficientNet-B0, AffectNet-8, ONNX) -----
EMOTION_MODEL = os.path.join(_HERE, "models", "enet_b0_8_best_vgaf.onnx")
FACE_CROP_PADDING = 0.15     # extra context around the face box before classifying
# Execution providers, best-first. Install `onnxruntime-directml` on Windows to
# use the GTX 1650 without CUDA; plain `onnxruntime` = CPU (also plenty fast).
PROVIDER_PRIORITY = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

# ----- Smoothing / stability -----
BOX_SMOOTHING = 0.55         # EMA factor for box position (higher = snappier)
PROB_SMOOTHING = 0.55        # EMA on the emotion probability vector per person
SWITCH_MARGIN = 0.06         # new emotion must beat the current one by this much (hysteresis)
TRACK_MAX_MISSES = 10        # inference cycles a track survives without a detection
IOU_MATCH_THRESHOLD = 0.25

# ----- Sarcasm mapping -----
# AffectNet includes CONTEMPT (the one-sided smirk) - the closest trained
# expression to sarcasm. sarcasm = contempt + this share of disgust;
# the remaining disgust goes to angry.
SARCASM_DISGUST_SHARE = 0.5
