"""
Fallback model fetcher. The models/ folder ships with the project, so you
normally never need this. Run `python download_models.py` only if the .onnx
files are missing or corrupted.
"""

import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "yunet_n_640_640.onnx":
        "https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_n_640_640.onnx",
    "enet_b0_8_best_vgaf.onnx":
        "https://raw.githubusercontent.com/av-savchenko/face-emotion-recognition/master/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx",
}


def main():
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    for name, url in MODELS.items():
        dest = os.path.join(HERE, "models", name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            print(f"[skip] {name} already present")
            continue
        print(f"[down] {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[ ok ] {name} ({os.path.getsize(dest)//1024} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
