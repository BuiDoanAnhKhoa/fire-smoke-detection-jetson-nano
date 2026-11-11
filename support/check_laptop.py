"""Read-only checks for an existing laptop Python environment. Installs nothing."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "736629455959a482400e68fdd93d7149f2a2b65b2dfb432f9c26598108b18a1c"


def main():
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.split()[0])
    try:
        import torch
        import cv2
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print("Missing or incompatible dependency:", exc)
        print("Select your existing working .venv, then try again.")
        return 1
    print("PyTorch:", torch.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("OpenCV:", cv2.__version__)
    weights = ROOT / "models/best.pt"
    if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != EXPECTED:
        print("Checkpoint missing or checksum differs. Extract the package again.")
        return 1
    model = YOLO(str(weights)).to("cpu")
    if model.names != {0: "fire", 1: "smoke"}:
        print("Unexpected classes:", model.names)
        return 1
    print("PASS: correct weights load on CPU; classes are fire and smoke.")
    print("Next: python predict_fire.py --source 0 --show")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
