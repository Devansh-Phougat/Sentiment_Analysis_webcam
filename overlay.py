"""
Overlay rendering: floating emoji + emotion label per person.

Emoji strategy
--------------
1. Try to rasterize real color emoji from a system font (Segoe UI Emoji on
   Windows, Apple Color Emoji on macOS, Noto Color Emoji on Linux) using
   Pillow's embedded_color support. Each emoji is rendered ONCE at startup
   into an RGBA sprite, then alpha-blended per frame (fast).
2. If no emoji font is found, fall back to procedurally drawn cartoon faces
   so the project still looks good anywhere.
"""

import math
import os
import sys
import time

import cv2
import numpy as np
import config

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# --------------------------------------------------------------------- #
#  Emoji sprite factory
# --------------------------------------------------------------------- #

def _find_emoji_font():
    candidates = []
    if sys.platform.startswith("win"):
        candidates.append(os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                                       "Fonts", "seguiemj.ttf"))
    elif sys.platform == "darwin":
        candidates.append("/System/Library/Fonts/Apple Color Emoji.ttc")
    candidates += [
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/noto-color-emoji/NotoColorEmoji.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _render_font_sprite(char, font_path, size):
    """Rasterize one emoji char to an RGBA numpy sprite, or None on failure."""
    try:
        # Noto Color Emoji (CBDT) only rasterizes at its native 109px size.
        native = 109 if "Noto" in font_path else size
        font = ImageFont.truetype(font_path, native)
        canvas = Image.new("RGBA", (native * 2, native * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((native // 2, native // 2), char, font=font, embedded_color=True)
        bbox = canvas.getbbox()
        if bbox is None:
            return None
        sprite = canvas.crop(bbox).resize((size, size), Image.LANCZOS)
        arr = np.array(sprite)                      # RGBA
        arr[..., :3] = arr[..., 2::-1]              # -> BGRA for OpenCV
        return arr
    except Exception:
        return None


def _draw_fallback_sprite(emotion, size):
    """Hand-drawn cartoon face (BGRA) — used only when no emoji font exists."""
    s = size
    img = np.zeros((s, s, 4), dtype=np.uint8)
    c = s // 2
    r = int(s * 0.46)
    face = (60, 200, 255, 255)  # warm yellow (BGRA)
    cv2.circle(img, (c, c), r, face, -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), r, (30, 120, 200, 255), max(2, s // 48), cv2.LINE_AA)

    eye_y = c - r // 4
    ex = r // 2
    er = max(2, s // 16)
    dark = (40, 40, 40, 255)

    def eyes(open_r=er):
        cv2.circle(img, (c - ex, eye_y), open_r, dark, -1, cv2.LINE_AA)
        cv2.circle(img, (c + ex, eye_y), open_r, dark, -1, cv2.LINE_AA)

    mouth_y = c + r // 3
    if emotion == "happy":
        eyes()
        cv2.ellipse(img, (c, mouth_y), (r // 2, r // 3), 0, 15, 165, dark, max(2, s // 32), cv2.LINE_AA)
    elif emotion == "sad":
        eyes()
        cv2.ellipse(img, (c, mouth_y + r // 3), (r // 2, r // 3), 0, 195, 345, dark, max(2, s // 32), cv2.LINE_AA)
        cv2.circle(img, (c + ex, eye_y + er * 2), er // 2, (230, 160, 60, 255), -1, cv2.LINE_AA)
    elif emotion == "angry":
        eyes()
        t = max(2, s // 32)
        cv2.line(img, (c - ex - er, eye_y - er * 2), (c - ex + er, eye_y - er), dark, t, cv2.LINE_AA)
        cv2.line(img, (c + ex + er, eye_y - er * 2), (c + ex - er, eye_y - er), dark, t, cv2.LINE_AA)
        cv2.ellipse(img, (c, mouth_y + r // 4), (r // 2, r // 4), 0, 200, 340, dark, t, cv2.LINE_AA)
    elif emotion == "surprise":
        eyes(er + s // 32)
        cv2.circle(img, (c, mouth_y), r // 4, dark, max(2, s // 32), cv2.LINE_AA)
    elif emotion == "fear":
        eyes(er + s // 32)
        cv2.ellipse(img, (c, mouth_y), (r // 3, r // 5), 0, 0, 360, dark, max(2, s // 32), cv2.LINE_AA)
        cv2.ellipse(img, (c, c), (r, r), 0, 220, 320, (200, 130, 60, 255), max(2, s // 24), cv2.LINE_AA)
    elif emotion == "sarcasm":
        # smirk: one raised brow + lopsided smile + half-lidded eyes
        t = max(2, s // 32)
        cv2.line(img, (c - ex - er, eye_y - er), (c - ex + er, eye_y - er), dark, t, cv2.LINE_AA)
        cv2.line(img, (c + ex - er, eye_y - er * 2), (c + ex + er, eye_y - er * 2), dark, t, cv2.LINE_AA)
        eyes(er)
        cv2.ellipse(img, (c + r // 6, mouth_y), (r // 2, r // 4), -15, 20, 150, dark, t, cv2.LINE_AA)
    else:  # neutral
        eyes()
        cv2.line(img, (c - r // 2, mouth_y), (c + r // 2, mouth_y), dark, max(2, s // 32), cv2.LINE_AA)
    return img


def build_sprites():
    """Pre-render one BGRA sprite per emotion. Returns (sprites, used_font)."""
    sprites, font_path = {}, _find_emoji_font() if _PIL_OK else None
    for emotion in config.EMOTIONS:
        sprite = None
        if font_path:
            sprite = _render_font_sprite(config.EMOJI[emotion], font_path,
                                         config.EMOJI_BASE_SIZE)
        if sprite is None:
            sprite = _draw_fallback_sprite(emotion, config.EMOJI_BASE_SIZE)
        sprites[emotion] = sprite
    return sprites, bool(font_path)


# --------------------------------------------------------------------- #
#  Compositing helpers
# --------------------------------------------------------------------- #

def _blend(frame, sprite, x, y):
    """Alpha-blend a BGRA sprite onto BGR frame at top-left (x, y), clipped."""
    fh, fw = frame.shape[:2]
    sh, sw = sprite.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + sw, fw), min(y + sh, fh)
    if x1 >= x2 or y1 >= y2:
        return
    sub = sprite[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = sub[..., 3:4].astype(np.float32) / 255.0
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (alpha * sub[..., :3] + (1 - alpha) * roi).astype(np.uint8)


def _rounded_pill(frame, cx, top_y, text, color):
    """Semi-transparent rounded label centered at cx."""
    font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (tw, th), _ = cv2.getTextSize(text, font, fscale, thick + 1)
    pad_x, pad_y = 12, 8
    w, h = tw + pad_x * 2, th + pad_y * 2
    x, y = int(cx - w / 2), int(top_y)
    fh, fw = frame.shape[:2]
    x = max(2, min(x, fw - w - 2))
    y = max(2, min(y, fh - h - 2))
    r = h // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), (25, 25, 25), -1)
    cv2.circle(overlay, (x + r, y + r), r, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.circle(overlay, (x + w - r, y + r), r, (25, 25, 25), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # colored accent dot + text
    cv2.circle(frame, (x + r, y + r), 4, color, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + pad_x + 6, y + h - pad_y - 2),
                font, fscale, (255, 255, 255), thick + 1, cv2.LINE_AA)


# --------------------------------------------------------------------- #
#  Public renderer
# --------------------------------------------------------------------- #

class OverlayRenderer:
    def __init__(self):
        self.sprites, self.using_font = build_sprites()
        self._cache = {}   # (emotion, size) -> resized sprite

    def _sprite(self, emotion, size):
        key = (emotion, size)
        if key not in self._cache:
            base = self.sprites[emotion]
            self._cache[key] = cv2.resize(base, (size, size),
                                          interpolation=cv2.INTER_AREA)
            if len(self._cache) > 256:
                self._cache.clear()
        return self._cache[key]

    def draw(self, frame, tracks, mode):
        now = time.time()
        for tr in tracks:
            x, y, w, h = (int(v) for v in tr.box)
            cx = x + w // 2
            color = config.COLORS.get(tr.emotion, (255, 255, 255))
            label = f"{tr.emotion.upper()}  {tr.score:.0%}"

            # subtle face outline so it's obvious who each overlay belongs to
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

            bob = int(config.BOB_AMPLITUDE
                      * math.sin(config.BOB_SPEED * now + tr.id * 1.7))

            if mode in ("emoji", "both"):
                size = max(40, min(140, int(w * 0.55)))
                # size snapped to a grid so the resize cache stays small
                size = int(round(size / 8) * 8)
                sprite = self._sprite(tr.emotion, size)
                ex = cx - size // 2
                ey = y - size - 14 + bob
                _blend(frame, sprite, ex, ey)
                text_top = ey + size + 4 if mode == "both" else None
            else:
                text_top = y - 34 + bob

            if mode in ("text", "both"):
                if mode == "both":
                    _rounded_pill(frame, cx, min(text_top, y - 30), label, color)
                else:
                    _rounded_pill(frame, cx, text_top, label, color)
