"""Train YOLOv8n, load its best checkpoint, then evaluate fire/smoke detection.
"""
import argparse
from datetime import datetime
from pathlib import Path

from support.metrics import save_metrics

ROOT = Path(__file__).resolve().parent


def train(name):
    import torch
    import yaml
    from ultralytics import YOLO

    # 1. Check the inputs and protect existing experiments from being overwritten.
    data = ROOT / "HomeFire/data_grouped.yaml"
    pretrained = ROOT / "models/yolov8n.pt"
    output = ROOT / "results/experiments" / name
    if output.exists():
        raise FileExistsError(f"Choose a new run name; this folder exists: {output}")
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    dataset = yaml.safe_load(data.read_text())
    for split in ("train", "val", "test"):
        if not Path(dataset[split]).is_file():
            raise FileNotFoundError(f"Missing {split} image list: {dataset[split]}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires a CUDA-enabled PyTorch installation and a compatible NVIDIA GPU.")

    # 2. Fine-tune the COCO-pretrained model
    model = YOLO(str(pretrained))
    model.train(
        data=str(data),
        epochs=1000,          # High ceiling; use early stopping
        time=8.0,
        patience=40,          # Stop after 40 epochs without better validation mAP.
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        amp=False,            # FP32 avoids GTX 1650 mixed-precision instability.
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        seed=42,
        deterministic=True,
        cache=False,
        pretrained=True,
        compile=False,
        close_mosaic=10,
        val=True,
        plots=True,
        save=True,            # Ultralytics saves best.pt and last.pt.
        save_period=5,        # Additional periodic checkpoints.
        project=str(output.parent),
        name=name,
        exist_ok=False,
    )
    best_path = Path(model.trainer.save_dir) / "weights/best.pt"
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Evaluate the selected best.pt, not the last training epoch.
    best = YOLO(str(best_path))
    evaluations = {}
    for split in ("val", "test"):
        evaluations[split] = best.val(
            data=str(data), split=split, imgsz=640, batch=4,
            device=0, workers=4, fraction=1.0,
            conf=0.001, iou=0.7, max_det=300,
            plots=True, save_json=False,
            project=str(best_path.parent.parent), name=f"evaluation_{split}",
            exist_ok=False,
        )

    # 4. Save a short report. Formatting lives in support/ to keep this flow clear.
    save_metrics(evaluations, best_path)
    print(f"Best model: {best_path}")
    print(f"Report: {best_path.parent.parent / 'REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=datetime.now().strftime("homefire_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("Use a single folder name, such as homefire_v2")
    train(args.name)


if __name__ == "__main__":
    main()
