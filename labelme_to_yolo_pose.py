import argparse
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Your semantic mapping / order (important!):
# point1=top-left, point2=mid-left, point3=bottom-left, point4=top-right, point5=mid-right, point6=bottom-right
KEYPOINT_LABELS = [f"point{i}" for i in range(1, 7)]
NUM_KPTS = len(KEYPOINT_LABELS)


@dataclass
class BoxInstance:
    xyxy: Tuple[float, float, float, float]
    area: float
    # label -> (x, y, v)
    kpts: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)


def rect_to_xyxy(rect_points: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in rect_points]
    ys = [p[1] for p in rect_points]
    return min(xs), min(ys), max(xs), max(ys)


def contains_xyxy(
    x1: float, y1: float, x2: float, y2: float, x: float, y: float, margin: float = 0.0
) -> bool:
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def yolo_norm_bbox(
    x1: float, y1: float, x2: float, y2: float, w: float, h: float
) -> Tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def find_pairs(ready_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for p in ready_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            jp = p.with_suffix(".json")
            if jp.exists():
                pairs.append((p, jp))
    return pairs


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(out_dir: Path, class_name: str) -> None:
    # Ultralytics dataset.yaml
    # flip_idx is 0-based indices for your left/right swap:
    # [p1,p2,p3,p4,p5,p6] -> [p4,p5,p6,p1,p2,p3]
    yaml_text = f"""path: {out_dir.as_posix()}
train: images/train
val: images/val

names:
  0: {class_name}

kpt_shape: [{NUM_KPTS}, 3]
flip_idx: [3, 4, 5, 0, 1, 2]
"""
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def write_summary_json(out_dir: Path, summary: dict) -> None:
    (out_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert LabelMe keypoint annotations (box + point1..point6) to YOLOv8 pose format."
    )
    ap.add_argument(
        "--ready",
        type=Path,
        default=Path("Ready"),
        help="Input folder containing .jpg/.png images and matching .json LabelMe files (default: Ready)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("dataset"),
        help="Output dataset folder (default: dataset)",
    )
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument(
        "--class-name",
        type=str,
        default="object",
        help="Class name to write into dataset.yaml (default: object)",
    )
    ap.add_argument(
        "--box-label",
        type=str,
        default="box",
        help='Rectangle label to treat as bounding box (default: "box")',
    )
    ap.add_argument(
        "--margin-px",
        type=float,
        default=3.0,
        help="Point-in-box tolerance in pixels when assigning points to boxes (default: 3.0)",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Delete output folder first if it exists (default: false)",
    )
    args = ap.parse_args()

    ready_dir: Path = args.ready
    out_dir: Path = args.out

    if not ready_dir.exists():
        raise SystemExit(f"Input folder not found: {ready_dir}")

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)

    pairs = find_pairs(ready_dir)
    if not pairs:
        raise SystemExit(f"No paired image+json found in: {ready_dir}")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_ratio)
    val_set = set([img.name for img, _ in pairs[:n_val]])

    for split in ["train", "val"]:
        safe_mkdir(out_dir / "images" / split)
        safe_mkdir(out_dir / "labels" / split)

    skipped_empty: List[str] = []
    skipped_no_boxes: List[str] = []
    missing_keypoints_lines: List[dict] = []

    written_images = 0
    written_instances = 0

    for img_path, json_path in pairs:
        split = "val" if img_path.name in val_set else "train"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        shapes = data.get("shapes", [])

        img_w = float(data["imageWidth"])
        img_h = float(data["imageHeight"])

        if not shapes:
            skipped_empty.append(json_path.name)
            continue

        # collect box instances
        boxes: List[BoxInstance] = []
        for s in shapes:
            if s.get("shape_type") != "rectangle":
                continue
            if s.get("label") != args.box_label:
                continue
            x1, y1, x2, y2 = rect_to_xyxy(s["points"])
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            boxes.append(BoxInstance(xyxy=(x1, y1, x2, y2), area=area))

        if not boxes:
            skipped_no_boxes.append(json_path.name)
            continue

        # assign points to smallest containing box
        for s in shapes:
            if s.get("shape_type") != "point":
                continue
            label = s.get("label")
            if label not in KEYPOINT_LABELS:
                continue
            x, y = s["points"][0]

            candidates: List[Tuple[float, int]] = []
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b.xyxy
                if contains_xyxy(x1, y1, x2, y2, x, y, margin=args.margin_px):
                    candidates.append((b.area, i))
            if not candidates:
                continue
            _, best_i = min(candidates, key=lambda t: t[0])
            boxes[best_i].kpts[label] = (float(x), float(y), 2.0)  # v=2 visible

        # write yolo pose lines, one per box
        lines: List[str] = []
        instance_missing_counts: List[int] = []

        for b in boxes:
            x1, y1, x2, y2 = b.xyxy
            cx, cy, bw, bh = yolo_norm_bbox(x1, y1, x2, y2, img_w, img_h)

            kpt_flat: List[float] = []
            missing = 0
            for lab in KEYPOINT_LABELS:
                if lab in b.kpts:
                    x, y, v = b.kpts[lab]
                    kpt_flat.extend([x / img_w, y / img_h, float(v)])
                else:
                    missing += 1
                    kpt_flat.extend([0.0, 0.0, 0.0])

            instance_missing_counts.append(missing)
            parts: List[float] = [0.0, cx, cy, bw, bh] + kpt_flat
            # class id must be int token, keep it separate
            line = "0 " + " ".join(f"{p:.6f}" for p in parts[1:])
            lines.append(line)

        out_img = out_dir / "images" / split / img_path.name
        out_lbl = out_dir / "labels" / split / (img_path.stem + ".txt")
        shutil.copy2(img_path, out_img)
        out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        written_images += 1
        written_instances += len(lines)

        if any(m > 0 for m in instance_missing_counts):
            missing_keypoints_lines.append(
                {
                    "file": json_path.name,
                    "instances": [
                        {"instance_index": i, "missing_keypoints": m}
                        for i, m in enumerate(instance_missing_counts)
                        if m > 0
                    ],
                }
            )

    write_dataset_yaml(out_dir, args.class_name)

    summary = {
        "input_ready": str(ready_dir),
        "output_dataset": str(out_dir),
        "pairs_found": len(pairs),
        "images_written": written_images,
        "instances_written": written_instances,
        "skipped_empty": sorted(skipped_empty),
        "skipped_no_boxes": sorted(skipped_no_boxes),
        "files_with_missing_keypoints": missing_keypoints_lines,
        "keypoint_order": KEYPOINT_LABELS,
        "flip_idx": [3, 4, 5, 0, 1, 2],
    }
    write_summary_json(out_dir, summary)

    print(f"Done. Wrote {written_images} images / {written_instances} instances to: {out_dir}")
    if skipped_empty:
        print(f"Skipped empty labels ({len(skipped_empty)}): {', '.join(skipped_empty)}")
    if skipped_no_boxes:
        print(f"Skipped no-box labels ({len(skipped_no_boxes)}): {', '.join(skipped_no_boxes)}")
    if missing_keypoints_lines:
        print(f"Warning: {len(missing_keypoints_lines)} files have instances with missing keypoints.")
        print(f"See: {out_dir / 'conversion_summary.json'}")


if __name__ == "__main__":
    main()


