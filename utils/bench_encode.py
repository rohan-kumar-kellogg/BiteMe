import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import time
from pathlib import Path as _Path
from PIL import Image
from models.vision import VisionEncoder

def bench(image_paths, n=100):
    enc = VisionEncoder()
    paths = image_paths[:n]

    # 1) PIL decode timing
    t0 = time.time()
    imgs = []
    for p in paths:
        imgs.append(Image.open(p).convert("RGB"))
    t_pil = time.time() - t0

    # 2) Full encode timing
    t1 = time.time()
    for p in paths:
        enc.encode_image(p)
    t_full = time.time() - t1

    print("device:", enc.device if hasattr(enc, "device") else "unknown")
    print(f"PIL decode: {t_pil:.2f}s for {n}  => {n/t_pil:.2f} imgs/s")
    print(f"Full encode: {t_full:.2f}s for {n} => {n/t_full:.2f} imgs/s")

if __name__ == "__main__":
    imgs = sorted([str(p) for p in Path("images").rglob("*.jpg")])
    bench(imgs, n=100)