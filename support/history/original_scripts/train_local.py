"""Local YOLOv8n training, atomic checkpoint mirrors, and automatic reporting."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".runtime" / "ultralytics"))


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def json_text(value):
    def convert(v):
        if isinstance(v, Path):
            return str(v)
        if hasattr(v, "item"):
            return v.item()
        if hasattr(v, "tolist"):
            return v.tolist()
        raise TypeError(type(v).__name__)
    return json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n"


def atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    temporary.replace(target)


def csv_rows(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def gpu_snapshot():
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


class RunArtifacts:
    def __init__(self, name, config):
        self.name = name
        self.run = ROOT / "runs" / "fire_detection" / name
        self.report = ROOT / "reports" / name
        self.backup = ROOT / "checkpoint_backups" / name
        self.report.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.state = {"status": "initializing", "created": now(), "run": str(self.run),
                      "requested_config": config, "evaluations": {}}
        self.last_record = 0
        self.last_heartbeat = 0
        self.epoch_batches = 0

    def write(self):
        self.state["updated"] = now()
        atomic_text(self.report / "status.json", json_text(self.state))
        atomic_text(self.report / "REPORT.md", self.markdown())

    def markdown(self):
        s = self.state
        lines = ["# HomeFire YOLOv8n training report", "",
                 f"- Status: **{s['status']}**", f"- Updated: {s.get('updated', now())}",
                 f"- Run: `{self.run}`", f"- Started: {s.get('training_started', 'pending')}",
                 f"- Stop reason: {s.get('stop_reason', 'pending')}",
                 f"- Epoch records: {s.get('epochs', 0)} (the timed final epoch may be partial)",
                 f"- Training-loop elapsed: {s.get('training_seconds', 0)/3600:.3f} hours",
                 f"- Training call including setup/final validation: {s.get('training_call_seconds', 0)/3600:.3f} hours",
                 f"- Best checkpoint epoch: {s.get('best_epoch', 'pending')}",
                 "", "## Configuration", "", "```json", json_text(self.config).strip(), "```", "",
                 "Training uses COCO pretrained YOLOv8n, with fire=0 and smoke=1. "
                 "The time limit takes precedence over the requested epoch count; Ultralytics adjusts "
                 "the epoch target and learning-rate schedule using measured epoch duration. "
                 "Early stopping monitors validation mAP50-95 with patience=40. "
                 "FP32 is used because this Ultralytics version flags GTX 1650 AMP instability.", "",
                 "## Dataset", "", "| Split | Images | Fire boxes | Smoke boxes | Empty labels |",
                 "|---|---:|---:|---:|---:|"]
        for split, d in s.get("dataset", {}).get("splits", {}).items():
            lines.append(f"| {split} | {d['images']} | {d['boxes'].get('0', 0)} | {d['boxes'].get('1', 0)} | {d['empty_labels']} |")
        lines += ["", "File integrity and annotation syntax were checked. Near-duplicate visual groups were "
                  "kept in one split using dHash distance <=8, 32px RGB RMSE <=18, and aspect ratio within 3%. "
                  "Seed 42 and group manifests make the new split reproducible. "
                  "Original splits had confirmed similar scenes in different splits. The new heuristic grouping "
                  "does not prove source/video independence or complete annotations. "
                  "Results describe this dataset; target-camera generalization needs separate evaluation.", ""]
        if s.get("latest_metrics"):
            lines += ["## Latest training validation", "", "```json", json_text(s["latest_metrics"]).strip(), "```", ""]
        for split, result in s.get("evaluations", {}).items():
            lines += [f"## Best checkpoint: {split}", "",
                      "| Class | Images containing class | Boxes | Precision | Recall | F1 | mAP50 | mAP50-95 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            a = result["aggregate"]
            lines.append(f"| all (class means) | — | — | {a['precision']:.4f} | {a['recall']:.4f} | {a['f1']:.4f} | {a['map50']:.4f} | {a['map50_95']:.4f} |")
            for c in result["per_class"]:
                lines.append(f"| {c['Class']} | {c['Images']} | {c['Instances']} | {c['Box-P']:.4f} | {c['Box-R']:.4f} | {c['Box-F1']:.4f} | {c['mAP50']:.4f} | {c['mAP50-95']:.4f} |")
            lines += ["", f"Measured milliseconds/image: `{result['speed_ms_per_image']}`.",
                      "These timings are on this desktop GPU during batched validation; they are not Jetson or webcam FPS.", ""]
        lines += ["Metrics are fractions from 0 to 1. Precision/recall/F1 use Ultralytics' "
                  "confidence operating point derived from the evaluation curves, not a fixed 0.5 threshold. "
                  "mAP50-95 averages AP over IoU thresholds 0.50–0.95. "
                  "Only validation determines checkpoint selection; test metrics are computed after training.", "",
                  "## Artifacts and recovery", "",
                  f"- Selected weights: `{self.run / 'weights/best.pt'}`",
                  f"- Epoch metrics and losses: `{self.run / 'results.csv'}`",
                  f"- Curves and confusion matrices: `{self.run}` and its `evaluation_*` directories",
                  f"- Resume checkpoint before optimizer stripping: `{self.backup / 'last_resume.pt'}`",
                  f"- Machine and environment: `{self.report / 'environment.json'}`",
                  f"- GPU telemetry: `{self.report / 'gpu.jsonl'}`",
                  "- Checkpoint mirrors are on the same physical disk; they do not protect against disk failure.", ""]
        if s.get("model"):
            lines += ["## Model details", "", "```json", json_text(s["model"]).strip(), "```", ""]
        if s.get("error"):
            lines += ["## Error", "", "```text", s["error"], "```", ""]
        return "\n".join(lines)

    def start(self, trainer):
        self.state.update(status="training", training_started=now(),
                          effective_batch=trainer.batch_size, amp=trainer.amp)
        self.write()

    def epoch_start(self, trainer):
        self.epoch_batches = 0

    def batch_end(self, trainer):
        self.epoch_batches += 1
        if time.monotonic() - self.last_heartbeat < 60:
            return
        self.last_heartbeat = time.monotonic()
        sample = {"time": now(), "epoch": trainer.epoch+1, "batch": self.epoch_batches,
                  "gpu": gpu_snapshot()}
        with (self.report / "gpu.jsonl").open("a") as f:
            f.write(json.dumps(sample) + "\n")
        atomic_text(self.report / "heartbeat.json", json_text(sample))

    def checkpoint(self, trainer):
        # This hook runs after a complete serialization, before final_eval strips optimizer state.
        for source, name in [(trainer.last, "last_resume.pt"), (trainer.best, "best.pt")]:
            if source.exists():
                atomic_copy(source, self.backup / name)

    def epoch_end(self, trainer):
        rows = csv_rows(Path(trainer.csv))
        if len(rows) == self.last_record:
            return  # final_eval invokes this hook again without adding an epoch
        self.last_record = len(rows)
        elapsed = time.time() - trainer.train_time_start
        self.state.update(epochs=len(rows), training_seconds=elapsed,
                          latest_metrics=rows[-1], effective_epoch_target=trainer.epochs,
                          effective_batch=trainer.batch_size,
                          last_epoch_batches=self.epoch_batches,
                          expected_epoch_batches=len(trainer.train_loader))
        if trainer.best_fitness == trainer.fitness:
            self.state["best_epoch"] = trainer.epoch + 1
        if trainer.stop:
            if elapsed >= (self.config.get("time") or math.inf)*3600:
                reason = f"training time budget reached ({self.config['time']} hours)"
            elif trainer.epoch+1-trainer.stopper.best_epoch >= trainer.stopper.patience:
                reason = "early stopping: no validation mAP50-95 improvement for 40 epochs"
            else:
                reason = "effective epoch target reached"
            self.state.update(stop_reason=reason, status="finishing training")
        self.write()
        atomic_copy(Path(trainer.csv), self.backup / "results.csv")
        print(f"ARTIFACTS: epoch={len(rows)}, best_epoch={self.state.get('best_epoch')}, "
              f"mAP50-95={trainer.fitness:.5f}, elapsed_hours={elapsed/3600:.3f}", flush=True)


def evaluate(artifacts, splits=("val", "test")):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    best = artifacts.run / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    model = YOLO(str(best))
    artifacts.state["model"] = {
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "gflops_at_640": get_flops(model.model, imgsz=640),
        "best_pt_bytes": best.stat().st_size,
        "best_pt_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
        "class_names": model.names}
    for split in splits:
        artifacts.state["status"] = f"evaluating {split}"
        artifacts.write()
        torch.cuda.empty_cache()
        result = model.val(data=str(ROOT / "HomeFire/data_grouped.yaml"), split=split, imgsz=640,
                           batch=4, device=0, workers=4, fraction=1.0, conf=0.001,
                           iou=0.7, max_det=300, plots=True, save_json=False,
                           project=str(artifacts.run), name=f"evaluation_{split}", exist_ok=True)
        value = {"aggregate": {"precision": float(result.box.mp), "recall": float(result.box.mr),
                               "f1": float(result.box.f1.mean()), "map50": float(result.box.map50),
                               "map50_95": float(result.box.map)},
                 "per_class": result.summary(decimals=6), "speed_ms_per_image": result.speed}
        artifacts.state["evaluations"][split] = value
        atomic_text(artifacts.report / f"metrics_{split}.json", json_text(value))
        artifacts.write()
    atomic_copy(best, artifacts.backup / "best_final.pt")
    artifacts.state.update(status="complete", finished=now())
    artifacts.write()
    atomic_copy(artifacts.report / "REPORT.md", artifacts.backup / "REPORT.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pilot", "train", "evaluate", "resume"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a single directory name")
    if args.batch < 1 or args.hours <= 0:
        parser.error("batch and hours must be positive")
    config = dict(data=str(ROOT / "HomeFire/data_grouped.yaml"), epochs=1000, time=args.hours,
                  imgsz=640, batch=args.batch, patience=40, device=0, workers=4,
                  amp=False, cache=False, optimizer="AdamW", lr0=0.001, lrf=0.01,
                  seed=42, deterministic=True, save=True, save_period=5, val=True,
                  plots=True, close_mosaic=10, pretrained=True, compile=False,
                  project=str(ROOT / "runs/fire_detection"), name=args.name, exist_ok=False)
    if args.mode == "pilot":
        config.update(epochs=2, time=None, fraction=0.1, close_mosaic=0)
    artifacts = RunArtifacts(args.name, config)
    if args.mode in ("evaluate", "resume"):
        statefile = artifacts.report / "status.json"
        if not statefile.exists():
            raise FileNotFoundError(f"No existing report state: {statefile}")
        artifacts.state = json.loads(statefile.read_text())
        artifacts.config = artifacts.state["requested_config"]
    elif artifacts.run.exists():
        raise FileExistsError(f"Refusing to overwrite {artifacts.run}")
    try:
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        import torch
        from ultralytics import YOLO
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
        if args.mode == "evaluate":
            evaluate(artifacts)
            return
        if args.mode != "resume":
            artifacts.state["dataset"] = json.loads((ROOT / "reports/preflight/grouped_audit.json").read_text())
            audit = artifacts.state["dataset"]
            if audit["errors"] or audit["cross_split_exact_duplicates"]:
                raise RuntimeError("Resolve dataset audit errors before training")
            environment = {"time": now(), "python": sys.version, "executable": sys.executable,
                           "platform": platform.platform(), "gpu": gpu_snapshot(),
                           "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                           "packages": {n: importlib.metadata.version(n) for n in
                                        ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "pillow", "PyYAML")}}
            atomic_text(artifacts.report / "environment.json", json_text(environment))
            frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            atomic_text(artifacts.report / "requirements-frozen.txt", frozen)
            for source in (ROOT / "train_local.py", ROOT / "HomeFire/data_grouped.yaml",
                           ROOT / "reports/preflight/grouped_audit.json"):
                atomic_copy(source, artifacts.report / "configuration" / source.name)
        artifacts.write()
        if args.mode == "resume":
            checkpoint = args.checkpoint or artifacts.backup / "last_resume.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model = YOLO(str(checkpoint))
            config = dict(resume=True, device=0, batch=args.batch, workers=4, time=args.hours,
                          save_dir=str(artifacts.run), data=str(ROOT / "HomeFire/data_grouped.yaml"))
            artifacts.state.setdefault("resumes", []).append({"at": now(), "checkpoint": str(checkpoint), "hours": args.hours})
            artifacts.config = {**artifacts.config, "time": args.hours, "batch": args.batch}
        else:
            model = YOLO(str(ROOT / "models/yolov8n.pt"))
        model.add_callback("on_train_start", artifacts.start)
        model.add_callback("on_train_epoch_start", artifacts.epoch_start)
        model.add_callback("on_train_batch_end", artifacts.batch_end)
        model.add_callback("on_model_save", artifacts.checkpoint)
        model.add_callback("on_fit_epoch_end", artifacts.epoch_end)
        started = time.monotonic()
        model.train(**config)
        artifacts.state["training_call_seconds"] = time.monotonic() - started
        if not (artifacts.run / "weights/best.pt").exists():
            raise RuntimeError("Training returned without best.pt")
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if args.mode == "pilot":
            evaluate(artifacts, splits=("val",))
            artifacts.state.update(status="pilot complete", finished=now())
            artifacts.write()
        else:
            evaluate(artifacts)
    except BaseException:
        artifacts.state.update(status="failed", error=traceback.format_exc())
        artifacts.write()
        raise


if __name__ == "__main__":
    main()
