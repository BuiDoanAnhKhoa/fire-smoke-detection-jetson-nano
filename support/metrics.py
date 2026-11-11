"""Save final best.pt evaluation results without adding reporting code to training."""
import hashlib
import json


def save_metrics(evaluations, best_path):
    output = best_path.parent.parent
    lines = ["# Best checkpoint results", "", f"Model: `{best_path}`", "",
             f"SHA256: `{hashlib.sha256(best_path.read_bytes()).hexdigest()}`", "",
             "| Split | Class | Precision | Recall | F1 | mAP50 | mAP50-95 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for split, result in evaluations.items():
        aggregate = dict(precision=float(result.box.mp), recall=float(result.box.mr),
                         f1=float(result.box.f1.mean()), map50=float(result.box.map50),
                         map50_95=float(result.box.map))
        per_class = result.summary(decimals=6)
        value = dict(aggregate=aggregate, per_class=per_class, speed_ms_per_image=result.speed)
        (output / f"metrics_{split}.json").write_text(
            json.dumps(value, indent=2, default=lambda v: v.item(), allow_nan=False))
        a = aggregate
        lines.append(f"| {split} | all (class means) | {a['precision']:.4f} | {a['recall']:.4f} | "
                     f"{a['f1']:.4f} | {a['map50']:.4f} | {a['map50_95']:.4f} |")
        for c in per_class:
            lines.append(f"| {split} | {c['Class']} | {c['Box-P']:.4f} | {c['Box-R']:.4f} | "
                         f"{c['Box-F1']:.4f} | {c['mAP50']:.4f} | {c['mAP50-95']:.4f} |")
    lines += ["", "Scores are fractions, not image classification accuracy. Precision/recall/F1 use "
              "the evaluator's confidence operating point, not a fixed confidence of 0.5.", "",
              "Checkpoint selection uses validation mAP50-95. Test evaluation happens after training. "
              "Inspect evaluation_val/ and evaluation_test/ for plots, and results.csv for epoch metrics.", "",
              "The dataset uses heuristic visual grouping; complete source-video independence is not established. "
              "Desktop evaluation timings do not establish webcam or Jetson FPS.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
