"""
Threaded emotion inference (v2).

Old stack : FER / mini-Xception, trained on FER2013 (48x48 grayscale) -> the
            "everything is neutral/sad/angry" bias, worse still because crops
            were taken from a 480px-wide downscaled frame.
New stack : HSEmotion EfficientNet-B0 (Savchenko), trained on AffectNet at
            224x224 COLOR, running on ONNX Runtime. Crops are cut from the
            FULL-resolution frame, expanded to a padded square, so the model
            finally sees real detail.

AffectNet's 8 classes include CONTEMPT - the unilateral smirk - which maps
naturally onto our 'sarcasm' label instead of the old hand-made heuristic.

AffectNet order : anger, contempt, disgust, fear, happiness, neutral,
                  sadness, surprise (alphabetical)
Project order   : see config.EMOTIONS
"""

import threading

import cv2
import numpy as np
import onnxruntime as ort

import config
from face_detector import FaceDetector

AFFECTNET_CLASSES = ["anger", "contempt", "disgust", "fear",
                     "happiness", "neutral", "sadness", "surprise"]
_A = {name: i for i, name in enumerate(AFFECTNET_CLASSES)}

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_PROVIDER_LABEL = {
    "DmlExecutionProvider": "DirectML(GPU)",
    "CUDAExecutionProvider": "CUDA(GPU)",
    "CPUExecutionProvider": "CPU",
}


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def map_to_seven(p8):
    """AffectNet-8 probabilities -> np.array aligned with config.EMOTIONS.

    sarcasm = contempt + share*disgust      (contempt IS the smirk class)
    angry   = anger    + (1-share)*disgust
    Mass is only moved around, so the result still sums to 1.
    """
    share = config.SARCASM_DISGUST_SHARE
    out = np.empty(len(config.EMOTIONS), dtype=np.float32)
    for i, emo in enumerate(config.EMOTIONS):
        if emo == "happy":
            out[i] = p8[_A["happiness"]]
        elif emo == "sad":
            out[i] = p8[_A["sadness"]]
        elif emo == "angry":
            out[i] = p8[_A["anger"]] + (1.0 - share) * p8[_A["disgust"]]
        elif emo == "surprise":
            out[i] = p8[_A["surprise"]]
        elif emo == "fear":
            out[i] = p8[_A["fear"]]
        elif emo == "neutral":
            out[i] = p8[_A["neutral"]]
        else:  # sarcasm
            out[i] = p8[_A["contempt"]] + share * p8[_A["disgust"]]
    return out


class EmotionEngine:
    """Background worker: feed it full-res frames, poll (box, probs) results."""

    def __init__(self):
        self.detector = FaceDetector()

        providers = [p for p in config.PROVIDER_PRIORITY
                     if p in ort.get_available_providers()]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(config.EMOTION_MODEL,
                                            sess_options=so,
                                            providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.img_size = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else 224
        self.can_batch = not isinstance(inp.shape[0], int)  # dynamic batch dim
        self.provider_label = _PROVIDER_LABEL.get(
            self.session.get_providers()[0], self.session.get_providers()[0])

        self._lock = threading.Lock()
        self._frame = None
        self._results = []            # list of (box, probs7)
        self._new_result = False
        self._running = True
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #

    def submit(self, frame):
        with self._lock:
            self._frame = frame
        self._event.set()

    def poll(self):
        with self._lock:
            fresh = self._new_result
            self._new_result = False
            return list(self._results), fresh

    def stop(self):
        self._running = False
        self._event.set()

    # ------------------------------------------------------------------ #

    def _crop(self, frame, box):
        """Padded SQUARE crop from the full-res frame -> normalized NCHW."""
        fh, fw = frame.shape[:2]
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        side = max(w, h) * (1.0 + 2.0 * config.FACE_CROP_PADDING)
        x1 = int(max(0, cx - side / 2))
        y1 = int(max(0, cy - side / 2))
        x2 = int(min(fw, cx + side / 2))
        y2 = int(min(fh, cy + side / 2))
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face = cv2.resize(face, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        return rgb.transpose(2, 0, 1)  # HWC -> CHW

    def _classify(self, tensors):
        """Run the model on a list of CHW tensors -> (N, 8) probabilities."""
        if self.can_batch:
            batch = np.stack(tensors).astype(np.float32)
            logits = self.session.run(None, {self.input_name: batch})[0]
        else:
            logits = np.concatenate([
                self.session.run(None, {self.input_name: t[None]})[0]
                for t in tensors
            ])
        return softmax(logits, axis=1)

    def _loop(self):
        while self._running:
            self._event.wait()
            self._event.clear()
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None:
                continue

            try:
                detections = self.detector.detect(frame)
                results = []
                if detections:
                    boxes = [b for b, _ in detections]
                    tensors, keep = [], []
                    for i, b in enumerate(boxes):
                        t = self._crop(frame, b)
                        if t is not None:
                            tensors.append(t)
                            keep.append(i)
                    if tensors:
                        probs8 = self._classify(tensors)
                        for j, i in enumerate(keep):
                            results.append((boxes[i], map_to_seven(probs8[j])))
            except Exception as exc:   # never kill the render loop
                print("inference error:", exc)
                results = []

            with self._lock:
                self._results = results
                self._new_result = True
