# Sentiment Analysis (v2 — ONNX stack)

Real-time facial emotion detection with floating emoji + text overlays for **every person in the frame simultaneously**.

## The 7 emotions

happy 😄 · sad 😢 · angry 😠 · surprise 😲 · fear 😨 · neutral 😐 · **sarcasm 😏**

Sarcasm now maps from AffectNet's trained **contempt** class (the one-sided smirk) plus a configurable share of disgust — a real learned expression, no longer a hand-made heuristic. It still can't read vocal/contextual sarcasm (nothing vision-only can), but a smirk/eye-roll face triggers it reliably.

## What changed in v2 and why

| Problem in v1 | Cause | v2 fix |
|---|---|---|
| Multiple rectangles on one face (worst up close) | FER's Haar cascade fires overlapping detections | **YuNet** face detector (`cv2.FaceDetectorYN`) with built-in NMS → one face, one box |
| Everything reads as neutral/sad/angry | mini-Xception trained on FER2013 (48×48 grayscale) + crops taken from a 480px downscaled frame | **HSEmotion EfficientNet-B0** trained on **AffectNet** at 224×224 color, crops cut from the **full-res** frame |
| Multi-face accuracy collapse | Both of the above compounding | Both fixes + batched inference (all faces in one model call) |
| Label flicker / stickiness | Hard label voting | EMA smoothing of the full **probability vector** + hysteresis margin |
| Dependency hell (TF/protobuf/pillow/moviepy) | TensorFlow-era stack | Everything runs on **ONNX Runtime** — tensorflow, fer, keras, moviepy all gone |

Both models ship in `models/` (16 MB total) — no downloads at first run.

## Setup

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Coming from v1? You can clean out the old stack — none of it is needed anymore:

```bat
pip uninstall -y tensorflow tensorflow-intel keras fer moviepy
```

You may also freely upgrade `protobuf` and `pillow` to their latest versions now; nothing in this project constrains them anymore.

## GPU on a GTX 1650 (optional, Windows)

CPU is already real-time. For extra headroom, use DirectML — no CUDA/cuDNN install needed:

```bat
pip uninstall -y onnxruntime
pip install onnxruntime-directml
```

The app auto-detects it (HUD will show `DirectML(GPU)` instead of `CPU`).

## Usage

```bat
python main.py                 # emoji + text (default)
python main.py --mode emoji
python main.py --mode text
python main.py --camera 1
```

Live keys: **E** emoji · **T** text · **B** both · **Q**/Esc quit.

## Architecture

```
camera loop (full FPS, never blocks)
 ├─ every 2nd frame ──► background thread:
 │                        YuNet detect @640px ─► padded square crops @ FULL res
 │                        ─► EfficientNet-B0 (batched) ─► 8 AffectNet probs
 │                        ─► remap to 7 emotions (contempt→sarcasm)
 ├─ IoU tracker: per-person EMA of box + probability vector, hysteresis switch
 └─ overlay: pre-rendered emoji sprites (bob animation) + rounded text pills
```

```
sentiment-analysis/
├── main.py              # capture loop, keys, HUD (shows FPS + inference device)
├── face_detector.py     # YuNet wrapper (NMS, full-res coordinate scaling)
├── emotion_engine.py    # ONNX Runtime inference thread, 8→7 mapping
├── tracker.py           # IoU tracking, prob-EMA smoothing, hysteresis
├── overlay.py           # emoji sprites, alpha blending, text pills
├── config.py            # every tunable
├── models/              # yunet_n_640_640.onnx + enet_b0_8_best_vgaf.onnx
├── download_models.py   # fallback fetcher (models already bundled)
└── requirements.txt
```

## Tuning (`config.py`)

- Faces missed at a distance → lower `MIN_FACE_SIZE` or `FACE_SCORE_THRESHOLD` (e.g. 0.55).
- False face boxes on clutter → raise `FACE_SCORE_THRESHOLD` (e.g. 0.75).
- Emotion feels laggy → raise `PROB_SMOOTHING` (snappier); flickery → lower it or raise `SWITCH_MARGIN`.
- Weak CPU + many people → `DETECT_EVERY_N = 3`.
- Sarcasm too eager/shy → adjust `SARCASM_DISGUST_SHARE` (0 = contempt only).

## Model credits

- Face detection: **YuNet** — Shiqi Yu et al., `libfacedetection` (BSD-3).
- Emotion: **HSEmotion** EfficientNet-B0 (AffectNet-8) — Andrey V. Savchenko (Apache-2.0).
