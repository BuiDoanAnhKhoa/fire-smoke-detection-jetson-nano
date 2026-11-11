"""Audit HomeFire without changing images, labels, or splits."""
import collections
import hashlib
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def audit(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"splits": {}, "errors": [], "boundary_warnings": []}
    exact, visual = collections.defaultdict(list), collections.defaultdict(list)
    manifest, samples = [], []
    for split in ("train", "val", "test"):
        images = sorted((ROOT / "HomeFire" / split / "images").glob("*.jpg"))
        labels = sorted((ROOT / "HomeFire" / split / "labels").glob("*.txt"))
        counts, empty = collections.Counter(), 0
        if {p.stem for p in images} != {p.stem for p in labels}:
            summary["errors"].append(f"{split}: image/label stems differ")
        for p in images:
            label = p.parent.parent / "labels" / (p.stem + ".txt")
            try:
                raw = p.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                exact[digest].append(str(p.relative_to(ROOT)))
                with Image.open(p) as im:
                    im.load()  # full decode, stronger than header verification
                    gray = list(im.convert("L").resize((9, 8)).get_flattened_data())
                    bits = sum(int(gray[y*9+x] > gray[y*9+x+1]) << (y*8+x)
                               for y in range(8) for x in range(8))
                    visual[bits].append(str(p.relative_to(ROOT)))
                rows = [r for r in label.read_text().splitlines() if r.strip()]
                empty += not rows
                for line in rows:
                    v = list(map(float, line.split()))
                    if (len(v) != 5 or not all(math.isfinite(x) for x in v)
                            or v[0] not in (0, 1) or not all(0 <= x <= 1 for x in v[1:])
                            or v[3] <= 0 or v[4] <= 0):
                        raise ValueError(f"invalid box: {line}")
                    counts[int(v[0])] += 1
                    _, x, y, w, h = v
                    if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
                        summary["boundary_warnings"].append(str(label.relative_to(ROOT)))
                manifest.append({"image": str(p.relative_to(ROOT)), "sha256": digest,
                                 "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest()})
            except Exception as exc:
                summary["errors"].append(f"{p}: {exc}")
        summary["splits"][split] = {"images": len(images), "labels": len(labels),
                                     "boxes": dict(counts), "empty_labels": empty}
        samples.extend(random.Random(42).sample(images, min(4, len(images))))
    crosses = lambda groups: [v for v in groups.values()
                             if len({Path(p).parts[1] for p in v}) > 1]
    summary["cross_split_exact_duplicates"] = crosses(exact)
    summary["cross_split_identical_dhash_candidates"] = crosses(visual)
    summary["limitations"] = ("Identical dHash values are review candidates, not proof of duplication. "
                              "This does not establish video/source independence or annotation completeness.")
    (destination / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    (destination / "dataset_audit.json").write_text(json.dumps(summary, indent=2))
    sheet = Image.new("RGB", (1200, 4 * 250), "#202020")
    for i, p in enumerate(samples):
        with Image.open(p) as original:
            im = original.convert("RGB")
        draw = ImageDraw.Draw(im)
        for row in (p.parent.parent / "labels" / (p.stem + ".txt")).read_text().splitlines():
            if not row.strip():
                continue
            c, x, y, w, h = map(float, row.split())
            color = "#ff4040" if c == 0 else "#40ffff"
            box = [(x-w/2)*im.width, (y-h/2)*im.height,
                   (x+w/2)*im.width, (y+h/2)*im.height]
            draw.rectangle(box, outline=color, width=max(3, im.width//250))
            draw.text((box[0]+4, box[1]+4), "fire" if c == 0 else "smoke", fill=color)
        im = ImageOps.contain(im, (400, 220))
        x, y = (i % 3)*400, (i//3)*250
        sheet.paste(im, (x, y+25))
        ImageDraw.Draw(sheet).text((x+5, y+5), f"{p.parts[-3]}/{p.name}", fill="white")
    sheet.save(destination / "label_review.jpg")
    print(json.dumps({**summary, "boundary_warnings": len(summary["boundary_warnings"]),
                      "cross_split_identical_dhash_candidates": len(summary["cross_split_identical_dhash_candidates"])}, indent=2))
    if summary["errors"] or summary["cross_split_exact_duplicates"]:
        raise RuntimeError("Dataset audit failed; inspect dataset_audit.json")
    return summary


if __name__ == "__main__":
    audit(ROOT / "support/reports" / "preflight")
