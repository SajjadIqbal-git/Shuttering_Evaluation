import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Keypoint meaning/order in your dataset:
# p1 = top-left, p2 = mid-left, p3 = bottom-left, p4 = top-right, p5 = mid-right, p6 = bottom-right
KPT_NAMES = ["point1", "point2", "point3", "point4", "point5", "point6"]


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_deg_from_vertical(vx: float, vy: float) -> float:
    """
    Returns absolute angle (degrees) from the vertical axis.
    0° means perfectly vertical. 90° means horizontal.
    """
    # angle from vertical: atan2(|vx|, |vy|)
    return math.degrees(math.atan2(abs(vx), abs(vy) if abs(vy) > 1e-9 else 1e-9))


def _angle_deg_from_horizontal(vx: float, vy: float) -> float:
    """
    Returns absolute angle (degrees) from the horizontal axis.
    0° means perfectly horizontal. 90° means vertical.
    """
    return math.degrees(math.atan2(abs(vy), abs(vx) if abs(vx) > 1e-9 else 1e-9))


def get_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    """
    Returns (width, height) for an image. Tries cv2 first (usually present with Ultralytics),
    then PIL as fallback.
    """
    try:
        import cv2  # type: ignore

        im = cv2.imread(str(image_path))
        if im is None:
            return None
        h, w = im.shape[:2]
        return int(w), int(h)
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return None


@dataclass
class DetectionGeom:
    image: str
    det_index: int
    # Ultralytics pose txt for your run doesn't include a detection confidence; keep for future compatibility.
    det_conf: float
    top_w_px: float
    mid_w_px: float
    bottom_w_px: float
    width_diff_px: float
    width_diff_pct: float
    centerline_angle_from_vertical_deg: float
    left_edge_angle_from_vertical_deg: float
    right_edge_angle_from_vertical_deg: float
    top_edge_angle_from_horizontal_deg: float
    pass_width: bool
    pass_angle: bool
    passed: bool
    notes: str


def parse_yolo_pose_line(
    tokens: List[str],
) -> Tuple[int, Tuple[float, float, float, float], List[Tuple[float, float, float]], float]:
    """
    Ultralytics `save_txt=True save_conf=True` pose format is typically:
      cls cx cy w h  (kpt1x kpt1y kpt1conf ... kpt6x kpt6y kpt6conf)  det_conf
    All coordinates are normalized [0..1].
    """
    cls = int(float(tokens[0]))
    floats = [float(t) for t in tokens[1:]]
    # For your current run, lines are: cls + 4 bbox + 6*3 keypoints = 1 + 4 + 18 = 23 tokens total
    if len(floats) < 4 + 18:
        raise ValueError(f"Not enough values for pose line: got {len(floats)} floats")

    bbox = tuple(floats[:4])  # (cx, cy, w, h)
    rest = floats[4:]

    # Some Ultralytics versions append det_conf at the end when save_conf=True.
    # If present, it's the last scalar after all keypoints.
    if len(rest) == 18:
        det_conf = 1.0
        kpt_vals = rest
    elif len(rest) == 19:
        det_conf = float(rest[-1])
        kpt_vals = rest[:-1]
    else:
        # If there are extra fields, try to interpret the last as det_conf and first 18 as keypoints
        det_conf = float(rest[-1])
        kpt_vals = rest[:18]

    kpts: List[Tuple[float, float, float]] = []
    for i in range(0, 18, 3):
        kpts.append((float(kpt_vals[i]), float(kpt_vals[i + 1]), float(kpt_vals[i + 2])))
    return cls, bbox, kpts, det_conf


def _fit_edge_angle_from_vertical(points: List[Tuple[float, float]]) -> float:
    """
    Fit x = a*y + b (least squares). For a vertical line, a ~ 0.
    Return absolute angle from vertical in degrees: atan(|a|).
    """
    if len(points) < 2:
        return float("nan")
    ys = [p[1] for p in points]
    xs = [p[0] for p in points]
    y_mean = sum(ys) / len(ys)
    x_mean = sum(xs) / len(xs)
    denom = sum((y - y_mean) ** 2 for y in ys)
    if denom < 1e-9:
        return float("nan")
    a = sum((y - y_mean) * (x - x_mean) for x, y in zip(xs, ys)) / denom
    return math.degrees(math.atan(abs(a)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute top/bottom widths and straightness angle from YOLOv8 pose predictions (save_txt output)."
    )
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("runs/pose/predict2"),
        help="Ultralytics predict folder (contains images and labels/). Default: runs/pose/predict2",
    )
    ap.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Override labels directory (default: <pred-dir>/labels)",
    )
    ap.add_argument(
        "--width-tol-px",
        type=float,
        default=10.0,
        help="Pass if |top_width - bottom_width| <= this many pixels (default: 10)",
    )
    ap.add_argument(
        "--width-tol-pct",
        type=float,
        default=0.03,
        help="Pass if |top-bottom|/max(top,bottom) <= this fraction (default: 0.03 = 3%%)",
    )
    ap.add_argument(
        "--use-mid-width",
        action="store_true",
        help="Also require mid width (p2-p5) to match top/bottom within tolerance (recommended).",
    )
    ap.add_argument(
        "--angle-tol-deg",
        type=float,
        default=3.0,
        help="Pass if centerline angle from vertical <= this many degrees (default: 3)",
    )
    ap.add_argument(
        "--kpt-min-conf",
        type=float,
        default=0.2,
        help="Treat a keypoint as valid only if its confidence >= this threshold (default: 0.2)",
    )
    ap.add_argument(
        "--one-per-image",
        action="store_true",
        help="Pick the best detection per image (by keypoint confidence + bbox area) and ignore others.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Where to write the CSV (default: <pred-dir>/geometry_report.csv)",
    )
    args = ap.parse_args()

    pred_dir: Path = args.pred_dir
    labels_dir: Path = args.labels_dir or (pred_dir / "labels")
    out_csv: Path = args.out_csv or (pred_dir / "geometry_report.csv")

    if not pred_dir.exists():
        raise SystemExit(f"pred-dir not found: {pred_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"labels-dir not found: {labels_dir}")

    txt_files = sorted(labels_dir.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in: {labels_dir}")

    rows: List[DetectionGeom] = []

    def best_det_key(det: dict) -> Tuple[float, float, int]:
        # Prefer more valid keypoints, higher mean kpt confidence, larger box area
        return (det["valid_kpts"], det["mean_kpt_conf"], det["area"])

    for txt_path in txt_files:
        stem = txt_path.stem
        # image should exist in pred_dir with common extensions; try jpg first.
        image_path = pred_dir / f"{stem}.jpg"
        if not image_path.exists():
            # try other extensions
            for ext in [".png", ".jpeg", ".webp", ".bmp"]:
                p2 = pred_dir / f"{stem}{ext}"
                if p2.exists():
                    image_path = p2
                    break

        size = get_image_size(image_path) if image_path.exists() else None
        if size is None:
            # Fall back to normalized geometry only (treat pixels as 1)
            w, h = 1, 1
            size_note = "missing_image_or_failed_read; using normalized units"
        else:
            w, h = size
            size_note = ""

        lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        parsed_dets: List[dict] = []

        for det_i, ln in enumerate(lines):
            tokens = ln.split()
            try:
                _, bbox, kpts, det_conf = parse_yolo_pose_line(tokens)
            except Exception as e:
                rows.append(
                    DetectionGeom(
                        image=image_path.name if image_path.exists() else f"{stem}.*",
                        det_index=det_i,
                        det_conf=0.0,
                        top_w_px=float("nan"),
                        mid_w_px=float("nan"),
                        bottom_w_px=float("nan"),
                        width_diff_px=float("nan"),
                        width_diff_pct=float("nan"),
                        centerline_angle_from_vertical_deg=float("nan"),
                        left_edge_angle_from_vertical_deg=float("nan"),
                        right_edge_angle_from_vertical_deg=float("nan"),
                        top_edge_angle_from_horizontal_deg=float("nan"),
                        pass_width=False,
                        pass_angle=False,
                        passed=False,
                        notes=f"parse_error: {e}",
                    )
                )
                continue

            cx, cy, bw, bh = bbox
            area = bw * bh
            mean_kpt_conf = sum(kc for _, _, kc in kpts) / max(len(kpts), 1)
            valid_kpts = sum(1 for _, _, kc in kpts if kc >= args.kpt_min_conf)
            parsed_dets.append(
                {
                    "det_index": det_i,
                    "bbox": bbox,
                    "kpts": kpts,
                    "det_conf": det_conf,
                    "area": area,
                    "mean_kpt_conf": mean_kpt_conf,
                    "valid_kpts": valid_kpts,
                }
            )

        if args.one_per_image and parsed_dets:
            parsed_dets.sort(key=best_det_key, reverse=True)
            parsed_dets = [parsed_dets[0]]

        for det in parsed_dets:
            det_i = det["det_index"]
            kpts = det["kpts"]
            det_conf = det["det_conf"]

            # Convert normalized to pixel units
            pts_xy: List[Optional[Tuple[float, float]]] = []
            bad = False
            notes = size_note
            for (x, y, kc) in kpts:
                if kc < args.kpt_min_conf or (x == 0.0 and y == 0.0):
                    pts_xy.append(None)
                else:
                    pts_xy.append((x * w, y * h))

            # Require the four corners for width + straightness: p1, p3, p4, p6
            p1, p2, p3, p4, p5, p6 = pts_xy
            if p1 is None or p4 is None or p3 is None or p6 is None:
                bad = True
                missing = [KPT_NAMES[i] for i, p in enumerate([p1, p2, p3, p4, p5, p6]) if p is None]
                notes = (notes + "; " if notes else "") + f"missing_keypoints:{'|'.join(missing)}"

            if bad:
                rows.append(
                    DetectionGeom(
                        image=image_path.name if image_path.exists() else f"{stem}.*",
                        det_index=det_i,
                        det_conf=float(det_conf),
                        top_w_px=float("nan"),
                        mid_w_px=float("nan"),
                        bottom_w_px=float("nan"),
                        width_diff_px=float("nan"),
                        width_diff_pct=float("nan"),
                        centerline_angle_from_vertical_deg=float("nan"),
                        left_edge_angle_from_vertical_deg=float("nan"),
                        right_edge_angle_from_vertical_deg=float("nan"),
                        top_edge_angle_from_horizontal_deg=float("nan"),
                        pass_width=False,
                        pass_angle=False,
                        passed=False,
                        notes=notes,
                    )
                )
                continue

            assert p1 is not None and p4 is not None and p3 is not None and p6 is not None

            top_w = _dist(p1, p4)
            bottom_w = _dist(p3, p6)
            mid_w = float("nan")
            if p2 is not None and p5 is not None:
                mid_w = _dist(p2, p5)
            diff_px = abs(top_w - bottom_w)
            denom = max(top_w, bottom_w, 1e-9)
            diff_pct = diff_px / denom

            # Straightness angle using centerline (top-mid to bottom-mid)
            top_mid = _mid(p1, p4)
            bottom_mid = _mid(p3, p6)
            vx = bottom_mid[0] - top_mid[0]
            vy = bottom_mid[1] - top_mid[1]
            angle_v = _angle_deg_from_vertical(vx, vy)

            # Edge angles from vertical (more robust than centerline alone)
            left_pts = [p for p in [p1, p2, p3] if p is not None]
            right_pts = [p for p in [p4, p5, p6] if p is not None]
            left_ang = _fit_edge_angle_from_vertical(left_pts) if len(left_pts) >= 2 else float("nan")
            right_ang = _fit_edge_angle_from_vertical(right_pts) if len(right_pts) >= 2 else float("nan")

            # Optional: top edge angle from horizontal (0°=perfectly horizontal)
            top_edge_vx = p4[0] - p1[0]
            top_edge_vy = p4[1] - p1[1]
            angle_top_edge = _angle_deg_from_horizontal(top_edge_vx, top_edge_vy)

            pass_width = (diff_px <= args.width_tol_px) or (diff_pct <= args.width_tol_pct)
            if args.use_mid_width and math.isfinite(mid_w):
                diff_tm = abs(top_w - mid_w)
                diff_bm = abs(bottom_w - mid_w)
                pass_width = pass_width and (
                    (diff_tm <= args.width_tol_px or (diff_tm / max(top_w, mid_w, 1e-9)) <= args.width_tol_pct)
                    and (diff_bm <= args.width_tol_px or (diff_bm / max(bottom_w, mid_w, 1e-9)) <= args.width_tol_pct)
                )

            # Use edge angles if available; else fall back to centerline
            edge_ok = True
            if math.isfinite(left_ang):
                edge_ok = edge_ok and (left_ang <= args.angle_tol_deg)
            if math.isfinite(right_ang):
                edge_ok = edge_ok and (right_ang <= args.angle_tol_deg)
            pass_angle = edge_ok and (angle_v <= (args.angle_tol_deg * 2.0))
            passed = pass_width and pass_angle

            rows.append(
                DetectionGeom(
                    image=image_path.name if image_path.exists() else f"{stem}.*",
                    det_index=det_i,
                    det_conf=float(det_conf),
                    top_w_px=float(top_w),
                    mid_w_px=float(mid_w) if math.isfinite(mid_w) else float("nan"),
                    bottom_w_px=float(bottom_w),
                    width_diff_px=float(diff_px),
                    width_diff_pct=float(diff_pct),
                    centerline_angle_from_vertical_deg=float(angle_v),
                    left_edge_angle_from_vertical_deg=float(left_ang) if math.isfinite(left_ang) else float("nan"),
                    right_edge_angle_from_vertical_deg=float(right_ang) if math.isfinite(right_ang) else float("nan"),
                    top_edge_angle_from_horizontal_deg=float(angle_top_edge),
                    pass_width=pass_width,
                    pass_angle=pass_angle,
                    passed=passed,
                    notes=notes,
                )
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(
            [
                "image",
                "det_index",
                "det_conf",
                "top_width_px",
                "mid_width_px",
                "bottom_width_px",
                "width_diff_px",
                "width_diff_pct",
                "centerline_angle_from_vertical_deg",
                "left_edge_angle_from_vertical_deg",
                "right_edge_angle_from_vertical_deg",
                "top_edge_angle_from_horizontal_deg",
                "pass_width",
                "pass_angle",
                "passed",
                "notes",
            ]
        )
        for r in rows:
            wcsv.writerow(
                [
                    r.image,
                    r.det_index,
                    f"{r.det_conf:.6f}",
                    f"{r.top_w_px:.3f}" if math.isfinite(r.top_w_px) else "",
                    f"{r.mid_w_px:.3f}" if math.isfinite(r.mid_w_px) else "",
                    f"{r.bottom_w_px:.3f}" if math.isfinite(r.bottom_w_px) else "",
                    f"{r.width_diff_px:.3f}" if math.isfinite(r.width_diff_px) else "",
                    f"{r.width_diff_pct:.6f}" if math.isfinite(r.width_diff_pct) else "",
                    f"{r.centerline_angle_from_vertical_deg:.3f}"
                    if math.isfinite(r.centerline_angle_from_vertical_deg)
                    else "",
                    f"{r.left_edge_angle_from_vertical_deg:.3f}"
                    if math.isfinite(r.left_edge_angle_from_vertical_deg)
                    else "",
                    f"{r.right_edge_angle_from_vertical_deg:.3f}"
                    if math.isfinite(r.right_edge_angle_from_vertical_deg)
                    else "",
                    f"{r.top_edge_angle_from_horizontal_deg:.3f}"
                    if math.isfinite(r.top_edge_angle_from_horizontal_deg)
                    else "",
                    int(r.pass_width),
                    int(r.pass_angle),
                    int(r.passed),
                    r.notes,
                ]
            )

    total = len(rows)
    passed = sum(1 for r in rows if r.passed)
    print(f"Wrote: {out_csv}")
    print(f"Detections: {total} | Passed: {passed} | Failed: {total - passed}")


if __name__ == "__main__":
    main()


