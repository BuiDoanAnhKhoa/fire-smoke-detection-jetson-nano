"""Create reproducible visual-group splits, preserving original images and labels.

Match dHash distance <=8, 32px RGB RMSE <=18, aspect ratio within 3%.
Connected components stay together. Source-video IDs remain unavailable.
"""
import collections
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "support/reports/preflight"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = ROOT / "HomeFire/splits"
    splits.mkdir(exist_ok=True)
    paths = sorted((ROOT / "HomeFire").glob("*/images/*.jpg"))
    hashes, thumbnails, aspects = [], [], []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            gray = np.asarray(im.convert("L").resize((9, 8)))
            bits = (gray[:, :-1] > gray[:, 1:]).ravel()
            hashes.append(sum(int(b) << i for i, b in enumerate(bits)))
            thumbnails.append(np.asarray(im.resize((32, 32)), dtype=np.float32).ravel())
            aspects.append(im.width / im.height)
    hashes = np.asarray(hashes, dtype=np.uint64)
    thumbnails = np.asarray(thumbnails)
    aspects = np.asarray(aspects)
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = []
    for i in range(len(paths)):
        distance = np.bitwise_count(hashes[i+1:] ^ hashes[i])
        candidates = np.flatnonzero(distance <= 8) + i + 1
        candidates = candidates[np.abs(aspects[candidates] / aspects[i] - 1) <= 0.03]
        if not len(candidates):
            continue
        rmse = np.sqrt(np.mean((thumbnails[candidates] - thumbnails[i])**2, axis=1))
        for j in candidates[rmse <= 18]:
            parent[find(int(j))] = find(i)
            edges.append((i, int(j)))
    groups = collections.defaultdict(list)
    for i in range(len(paths)):
        groups[find(i)].append(i)
    grouped = list(groups.values())
    rng = random.Random(42)
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)
    targets = {"train": .6*len(paths), "val": .2*len(paths), "test": .2*len(paths)}
    assigned = {s: [] for s in targets}
    records, image_split = [], {}
    for group_id, indices in enumerate(grouped):
        split = min(targets, key=lambda s: len(assigned[s])/targets[s])
        for i in indices:
            assigned[split].append(paths[i])
            image_split[i] = split
        records.append({"group": group_id, "split": split,
                        "images": [str(paths[i].relative_to(ROOT)) for i in indices]})
    assert all(image_split[i] == image_split[j] for i, j in edges)
    summary = {"splits": {}, "errors": [], "cross_split_exact_duplicates": [],
               "method": "connected components: dHash <=8, RGB thumbnail RMSE <=18, aspect ratio within 3%; seed 42",
               "groups": len(grouped), "largest_group": max(map(len, grouped)),
               "matched_pairs": len(edges), "detected_cross_split_pairs_after_grouping": 0,
               "limitations": "Heuristic visual grouping; unknown source-video IDs. Residual scene overlap may remain."}
    for split, images in assigned.items():
        counts, empty = collections.Counter(), 0
        for p in images:
            rows = (p.parent.parent / "labels" / (p.stem+".txt")).read_text().splitlines()
            rows = [r for r in rows if r.strip()]
            empty += not rows
            for row in rows:
                counts[int(row.split()[0])] += 1
        if len(images) < 500 or any(counts[c] < 100 for c in (0, 1)):
            raise RuntimeError(f"Insufficient images/classes in {split}")
        manifest = "".join(str(p)+"\n" for p in sorted(images))
        (OUT / f"grouped_{split}.txt").write_text(manifest)
        (splits / f"{split}.txt").write_text(manifest)
        summary["splits"][split] = {"images": len(images), "labels": len(images),
                                    "boxes": dict(counts), "empty_labels": empty,
                                    "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest()}
    (OUT / "grouped_audit.json").write_text(json.dumps(summary, indent=2))
    (OUT / "visual_groups.json").write_text(json.dumps(records, indent=2))
    config = {"path": str(ROOT / "HomeFire"),
              **{s: str(splits / f"{s}.txt") for s in targets},
              "names": {0: "fire", 1: "smoke"}}
    (ROOT / "HomeFire/data_grouped.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    review = rng.sample(edges, min(8, len(edges)))
    sheet = Image.new("RGB", (1000, len(review)*230), "#202020")
    for row, pair in enumerate(review):
        for col, i in enumerate(pair):
            with Image.open(paths[i]) as im:
                im = ImageOps.contain(im.convert("RGB"), (500, 200))
            sheet.paste(im, (col*500, row*230+25))
            ImageDraw.Draw(sheet).text((col*500+5,row*230+5),
                                      str(paths[i].relative_to(ROOT)), fill="white")
    sheet.save(OUT / "grouping_review.jpg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
