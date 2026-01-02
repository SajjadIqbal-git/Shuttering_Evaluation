import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Keypoint meaning/order:
# p1=top-left, p2=mid-left, p3=bottom-left, p4=top-right, p5=mid-right, p6=bottom-right
KPT_NAMES = ["point1", "point2", "point3", "point4", "point5", "point6"]


def parse_pose_txt_line(
    tokens: List[str],
) -> Tuple[int, Tuple[float, float, float, float], List[Tuple[float, float, float]]]:
    """
    Expected for your run: cls cx cy w h k1x k1y k1c ... k6x k6y k6c (all normalized).
    Total tokens: 1 + 4 + 18 = 23
    """
    if len(tokens) < 1 + 4 + 18:
        raise ValueError(f"Bad token length: {len(tokens)}")
    cls = int(float(tokens[0]))
    vals = [float(x) for x in tokens[1:]]
    bbox = tuple(vals[:4])  # cx,cy,w,h (normalized)
    k = vals[4 : 4 + 18]
    kpts = [(k[i], k[i + 1], k[i + 2]) for i in range(0, 18, 3)]
    return cls, bbox, kpts


def load_best_detection(txt_path: Path, kpt_min_conf: float) -> Optional[Dict]:
    lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None

    best = None
    best_key = None
    for i, ln in enumerate(lines):
        cls, bbox, kpts = parse_pose_txt_line(ln.split())
        cx, cy, bw, bh = bbox
        area = bw * bh
        mean_kpt_conf = sum(kc for _, _, kc in kpts) / len(kpts)
        valid_kpts = sum(1 for _, _, kc in kpts if kc >= kpt_min_conf)
        key = (valid_kpts, mean_kpt_conf, area)
        if best is None or key > best_key:
            best = {"det_index": i, "cls": cls, "bbox": bbox, "kpts": kpts, "key": key}
            best_key = key
    return best


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def angle_from_vertical_deg(vx: float, vy: float) -> float:
    return math.degrees(math.atan2(abs(vx), abs(vy) if abs(vy) > 1e-9 else 1e-9))


def angle_from_horizontal_deg(vx: float, vy: float) -> float:
    return math.degrees(math.atan2(abs(vy), abs(vx) if abs(vx) > 1e-9 else 1e-9))


def fit_edge_angle_from_vertical(points: List[Tuple[float, float]]) -> float:
    """
    Fit x = a*y + b. For vertical edges, a~0 so angle~0.
    Returns abs(angle) from vertical in degrees.
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


def get_image_size(image_path: Path) -> Tuple[int, int]:
    import cv2  # type: ignore

    im = cv2.imread(str(image_path))
    if im is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w = im.shape[:2]
    return int(w), int(h)


def compute_metrics(
    w: int,
    h: int,
    kpts_norm: List[Tuple[float, float, float]],
    kpt_min_conf: float,
    width_tol_px: float,
    width_tol_pct: float,
    angle_tol_deg: float,
) -> Dict:
    # Convert normalized -> px, filter by conf
    pts: List[Optional[Tuple[float, float]]] = []
    confs: List[float] = []
    for x, y, kc in kpts_norm:
        confs.append(kc)
        if kc < kpt_min_conf or (x == 0.0 and y == 0.0):
            pts.append(None)
        else:
            pts.append((x * w, y * h))

    p1, p2, p3, p4, p5, p6 = pts
    missing = [KPT_NAMES[i] for i, p in enumerate(pts) if p is None]

    out: Dict = {
        "top_w": float("nan"),
        "mid_w": float("nan"),
        "bottom_w": float("nan"),
        "width_diff_px": float("nan"),
        "width_diff_pct": float("nan"),
        "bottom_over_top": float("nan"),
        "angle_center_v": float("nan"),
        "angle_left_v": float("nan"),
        "angle_right_v": float("nan"),
        "angle_top_h": float("nan"),
        "passed": False,
        "pass_width": False,
        "pass_angle": False,
        "missing": missing,
        "pts_px": pts,
        "kpt_confs": confs,
    }

    if p1 is None or p4 is None or p3 is None or p6 is None:
        return out

    top_w = dist(p1, p4)
    bottom_w = dist(p3, p6)
    out["top_w"] = top_w
    out["bottom_w"] = bottom_w

    if p2 is not None and p5 is not None:
        out["mid_w"] = dist(p2, p5)

    diff_px = abs(top_w - bottom_w)
    diff_pct = diff_px / max(top_w, bottom_w, 1e-9)
    out["width_diff_px"] = diff_px
    out["width_diff_pct"] = diff_pct
    out["bottom_over_top"] = bottom_w / max(top_w, 1e-9)

    # Centerline
    top_mid = mid(p1, p4)
    bottom_mid = mid(p3, p6)
    out["angle_center_v"] = angle_from_vertical_deg(bottom_mid[0] - top_mid[0], bottom_mid[1] - top_mid[1])

    # Edge line-fitting
    left_pts = [p for p in [p1, p2, p3] if p is not None]
    right_pts = [p for p in [p4, p5, p6] if p is not None]
    if len(left_pts) >= 2:
        out["angle_left_v"] = fit_edge_angle_from_vertical(left_pts)
    if len(right_pts) >= 2:
        out["angle_right_v"] = fit_edge_angle_from_vertical(right_pts)

    # Top edge
    out["angle_top_h"] = angle_from_horizontal_deg(p4[0] - p1[0], p4[1] - p1[1])

    pass_width = (diff_px <= width_tol_px) or (diff_pct <= width_tol_pct)
    pass_angle = (out["angle_center_v"] <= angle_tol_deg)
    if math.isfinite(out["angle_left_v"]):
        pass_angle = pass_angle and (out["angle_left_v"] <= angle_tol_deg)
    if math.isfinite(out["angle_right_v"]):
        pass_angle = pass_angle and (out["angle_right_v"] <= angle_tol_deg)

    out["pass_width"] = pass_width
    out["pass_angle"] = pass_angle
    out["passed"] = bool(pass_width and pass_angle)
    return out


def draw_panel(img, panel_w: int, lines: List[Tuple[str, Tuple[int, int, int]]]):
    import cv2  # type: ignore

    h, w = img.shape[:2]
    canvas = cv2.copyMakeBorder(img, 0, 0, 0, panel_w, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    x0 = w + 15
    y = 35
    for text, color in lines:
        cv2.putText(canvas, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        y += 35
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Annotate images using YOLOv8-pose txt outputs with PASS/FAIL + widths + angles.")
    ap.add_argument("--pred-dir", type=Path, default=Path("runs/pose/predict2"), help="Folder with predicted images.")
    ap.add_argument("--labels-dir", type=Path, default=None, help="Folder with .txt labels (default: <pred-dir>/labels).")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output folder (default: <pred-dir>/annotated).")
    ap.add_argument("--width-tol-px", type=float, default=10.0, help="Width tolerance in px (default 10).")
    ap.add_argument("--width-tol-pct", type=float, default=0.08, help="Width tolerance in ratio (default 0.08=8%%).")
    ap.add_argument("--angle-tol-deg", type=float, default=5.0, help="Angle tolerance in degrees (default 5).")
    ap.add_argument("--kpt-min-conf", type=float, default=0.25, help="Min keypoint confidence (default 0.25).")
    ap.add_argument("--panel-w", type=int, default=360, help="Side panel width in pixels (default 360).")
    args = ap.parse_args()

    pred_dir: Path = args.pred_dir
    labels_dir: Path = args.labels_dir or (pred_dir / "labels")
    out_dir: Path = args.out_dir or (pred_dir / "annotated")
    out_dir.mkdir(parents=True, exist_ok=True)

    import cv2  # type: ignore

    images = sorted([p for p in pred_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".webp", ".bmp"}])
    if not images:
        raise SystemExit(f"No images found in {pred_dir}")

    passed = 0
    total = 0

    for img_path in images:
        txt_path = labels_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        det = load_best_detection(txt_path, args.kpt_min_conf) if txt_path.exists() else None
        if det is None:
            status = "NO DETECTION"
            color = (0, 165, 255)
            lines = [
                (status, color),
                (img_path.name, (200, 200, 200)),
            ]
            out = draw_panel(img, args.panel_w, lines)
            cv2.imwrite(str(out_dir / img_path.name), out)
            continue

        metrics = compute_metrics(
            w=w,
            h=h,
            kpts_norm=det["kpts"],
            kpt_min_conf=args.kpt_min_conf,
            width_tol_px=args.width_tol_px,
            width_tol_pct=args.width_tol_pct,
            angle_tol_deg=args.angle_tol_deg,
        )

        total += 1
        if metrics["passed"]:
            passed += 1

        # Draw keypoints/edges on the image
        pts = metrics["pts_px"]
        def draw_pt(p, c):
            if p is None:
                return
            cv2.circle(img, (int(p[0]), int(p[1])), 4, c, -1, lineType=cv2.LINE_AA)

        # corners emphasize
        draw_pt(pts[0], (0, 255, 255))  # p1
        draw_pt(pts[3], (0, 255, 255))  # p4
        draw_pt(pts[2], (0, 255, 255))  # p3
        draw_pt(pts[5], (0, 255, 255))  # p6
        # mids
        draw_pt(pts[1], (255, 255, 0))  # p2
        draw_pt(pts[4], (255, 255, 0))  # p5

        def draw_line(a, b, c, t=2):
            if a is None or b is None:
                return
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, t, lineType=cv2.LINE_AA)

        draw_line(pts[0], pts[3], (255, 0, 255), 2)  # top
        draw_line(pts[2], pts[5], (255, 0, 255), 2)  # bottom
        draw_line(pts[0], pts[2], (0, 255, 0), 2)  # left
        draw_line(pts[3], pts[5], (0, 255, 0), 2)  # right

        status = "PASSED" if metrics["passed"] else "FAILED"
        status_color = (0, 220, 0) if metrics["passed"] else (0, 0, 255)
        lines = [
            (status, status_color),
            (img_path.name, (200, 200, 200)),
            (f"Top width:    {metrics['top_w']:.1f}px", (230, 230, 230)) if math.isfinite(metrics["top_w"]) else ("Top width:    -", (230, 230, 230)),
            (f"Mid width:    {metrics['mid_w']:.1f}px", (230, 230, 230)) if math.isfinite(metrics["mid_w"]) else ("Mid width:    -", (230, 230, 230)),
            (f"Bottom width: {metrics['bottom_w']:.1f}px", (230, 230, 230)) if math.isfinite(metrics["bottom_w"]) else ("Bottom width: -", (230, 230, 230)),
            (f"Diff: {metrics['width_diff_px']:.1f}px ({metrics['width_diff_pct']*100:.1f}%)", (230, 230, 230)) if math.isfinite(metrics["width_diff_px"]) else ("Diff: -", (230, 230, 230)),
            (f"Ratio (B/T): {metrics['bottom_over_top']:.3f}", (230, 230, 230)) if math.isfinite(metrics["bottom_over_top"]) else ("Ratio (B/T): -", (230, 230, 230)),
            (f"Angle(v): {metrics['angle_center_v']:.2f}°", (230, 230, 230)) if math.isfinite(metrics["angle_center_v"]) else ("Angle(v): -", (230, 230, 230)),
            (f"Left edge:  {metrics['angle_left_v']:.2f}°", (230, 230, 230)) if math.isfinite(metrics["angle_left_v"]) else ("Left edge:  -", (230, 230, 230)),
            (f"Right edge: {metrics['angle_right_v']:.2f}°", (230, 230, 230)) if math.isfinite(metrics["angle_right_v"]) else ("Right edge: -", (230, 230, 230)),
        ]
        if metrics["missing"]:
            lines.append((f"Missing: {','.join(metrics['missing'])}", (0, 165, 255)))

        out = draw_panel(img, args.panel_w, lines)
        cv2.imwrite(str(out_dir / img_path.name), out)

    print(f"Annotated images saved to: {out_dir}")
    if total:
        print(f"Summary (best det per image): {passed}/{total} passed ({passed/total*100:.1f}%)")


if __name__ == "__main__":
    main()


