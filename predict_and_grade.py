import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# Keypoint meaning/order:
# p1=top-left, p2=mid-left, p3=bottom-left, p4=top-right, p5=mid-right, p6=bottom-right
KPT_NAMES = ["point1", "point2", "point3", "point4", "point5", "point6"]


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def angle_from_vertical_deg(vx: float, vy: float) -> float:
    # 0° is vertical
    return math.degrees(math.atan2(abs(vx), abs(vy) if abs(vy) > 1e-9 else 1e-9))


def angle_from_horizontal_deg(vx: float, vy: float) -> float:
    # 0° is horizontal
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


@dataclass
class Metrics:
    passed: bool
    pass_width: bool
    pass_angle: bool
    top_w: float
    mid_w: float
    bottom_w: float
    width_diff_px: float
    width_diff_pct: float
    bottom_over_top: float
    angle_center_v: float
    angle_left_v: float
    angle_right_v: float
    angle_top_h: float
    missing: List[str]


def compute_metrics(
    pts_px: List[Optional[Tuple[float, float]]],
    width_tol_px: float,
    width_tol_pct: float,
    angle_tol_deg: float,
    use_mid_width: bool,
    width_mode: str,
) -> Metrics:
    p1, p2, p3, p4, p5, p6 = pts_px
    missing = [KPT_NAMES[i] for i, p in enumerate(pts_px) if p is None]

    if p1 is None or p4 is None or p3 is None or p6 is None:
        return Metrics(
            passed=False,
            pass_width=False,
            pass_angle=False,
            top_w=float("nan"),
            mid_w=float("nan"),
            bottom_w=float("nan"),
            width_diff_px=float("nan"),
            width_diff_pct=float("nan"),
            bottom_over_top=float("nan"),
            angle_center_v=float("nan"),
            angle_left_v=float("nan"),
            angle_right_v=float("nan"),
            angle_top_h=float("nan"),
            missing=missing,
        )

    top_w = dist(p1, p4)
    bottom_w = dist(p3, p6)
    mid_w = float("nan")
    if p2 is not None and p5 is not None:
        mid_w = dist(p2, p5)

    diff_px = abs(top_w - bottom_w)
    diff_pct = diff_px / max(top_w, bottom_w, 1e-9)
    bottom_over_top = bottom_w / max(top_w, 1e-9)

    # Centerline angle
    top_mid = mid(p1, p4)
    bottom_mid = mid(p3, p6)
    angle_center_v = angle_from_vertical_deg(bottom_mid[0] - top_mid[0], bottom_mid[1] - top_mid[1])

    # Edge line-fit angles
    left_pts = [p for p in [p1, p2, p3] if p is not None]
    right_pts = [p for p in [p4, p5, p6] if p is not None]
    angle_left_v = fit_edge_angle_from_vertical(left_pts) if len(left_pts) >= 2 else float("nan")
    angle_right_v = fit_edge_angle_from_vertical(right_pts) if len(right_pts) >= 2 else float("nan")

    # Top edge horizontalness
    angle_top_h = angle_from_horizontal_deg(p4[0] - p1[0], p4[1] - p1[1])

    # Width pass criteria:
    # - px: use absolute pixel tolerance (only valid if distance is fixed)
    # - pct: use relative tolerance (recommended when distance varies)
    # - either: pass if either check passes
    if width_mode == "px":
        pass_width = diff_px <= width_tol_px
    elif width_mode == "pct":
        pass_width = diff_pct <= width_tol_pct
    else:
        pass_width = (diff_px <= width_tol_px) or (diff_pct <= width_tol_pct)
    if use_mid_width and math.isfinite(mid_w):
        diff_tm = abs(top_w - mid_w)
        diff_bm = abs(bottom_w - mid_w)
        pass_width = pass_width and (
            (diff_tm <= width_tol_px or (diff_tm / max(top_w, mid_w, 1e-9)) <= width_tol_pct)
            and (diff_bm <= width_tol_px or (diff_bm / max(bottom_w, mid_w, 1e-9)) <= width_tol_pct)
        )

    pass_angle = angle_center_v <= angle_tol_deg
    if math.isfinite(angle_left_v):
        pass_angle = pass_angle and (angle_left_v <= angle_tol_deg)
    if math.isfinite(angle_right_v):
        pass_angle = pass_angle and (angle_right_v <= angle_tol_deg)

    passed = pass_width and pass_angle
    return Metrics(
        passed=passed,
        pass_width=pass_width,
        pass_angle=pass_angle,
        top_w=top_w,
        mid_w=mid_w,
        bottom_w=bottom_w,
        width_diff_px=diff_px,
        width_diff_pct=diff_pct,
        bottom_over_top=bottom_over_top,
        angle_center_v=angle_center_v,
        angle_left_v=angle_left_v,
        angle_right_v=angle_right_v,
        angle_top_h=angle_top_h,
        missing=missing,
    )


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


def pick_best_detection(kpts_xy: List[List[Tuple[float, float]]], kpts_conf: List[List[float]], box_xyxy: List[Tuple[float, float, float, float]], kpt_min_conf: float):
    best_i = None
    best_key = None
    for i in range(len(kpts_xy)):
        valid = sum(1 for c in kpts_conf[i] if c >= kpt_min_conf)
        mean_c = sum(kpts_conf[i]) / max(len(kpts_conf[i]), 1)
        x1, y1, x2, y2 = box_xyxy[i]
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        key = (valid, mean_c, area)
        if best_key is None or key > best_key:
            best_key = key
            best_i = i
    return best_i


def process_frame(
    frame_bgr,
    model,
    imgsz: int,
    conf: float,
    kpt_min_conf: float,
    width_tol_px: float,
    width_tol_pct: float,
    angle_tol_deg: float,
    use_mid_width: bool,
    width_mode: str,
    panel_w: int,
) -> Tuple:
    import cv2  # type: ignore

    res = model.predict(frame_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]

    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        lines = [("NO DETECTION", (0, 165, 255))]
        out = draw_panel(frame_bgr, panel_w, lines)
        return out, None

    # keypoints.xy: (N, K, 2) in pixel coords
    # keypoints.conf: (N, K) in [0..1]
    kxy = res.keypoints.xy.cpu().numpy().tolist()
    kconf = res.keypoints.conf.cpu().numpy().tolist() if res.keypoints.conf is not None else [[1.0] * 6 for _ in kxy]
    boxes = res.boxes.xyxy.cpu().numpy().tolist()
    boxes = [tuple(map(float, b)) for b in boxes]

    best_i = pick_best_detection(kxy, kconf, boxes, kpt_min_conf)
    if best_i is None:
        lines = [("NO VALID DETECTION", (0, 165, 255))]
        out = draw_panel(frame_bgr, panel_w, lines)
        return out, None

    pts_px: List[Optional[Tuple[float, float]]] = []
    for j in range(6):
        x, y = kxy[best_i][j]
        c = kconf[best_i][j]
        if c < kpt_min_conf:
            pts_px.append(None)
        else:
            pts_px.append((float(x), float(y)))

    m = compute_metrics(pts_px, width_tol_px, width_tol_pct, angle_tol_deg, use_mid_width, width_mode)

    # Draw box & edges
    x1, y1, x2, y2 = boxes[best_i]
    cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 0), 2)

    def draw_pt(p, c):
        if p is None:
            return
        cv2.circle(frame_bgr, (int(p[0]), int(p[1])), 4, c, -1, lineType=cv2.LINE_AA)

    def draw_line(a, b, c, t=2):
        if a is None or b is None:
            return
        cv2.line(frame_bgr, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), c, t, lineType=cv2.LINE_AA)

    p1, p2, p3, p4, p5, p6 = pts_px
    draw_pt(p1, (0, 255, 255))
    draw_pt(p4, (0, 255, 255))
    draw_pt(p3, (0, 255, 255))
    draw_pt(p6, (0, 255, 255))
    draw_pt(p2, (255, 255, 0))
    draw_pt(p5, (255, 255, 0))

    draw_line(p1, p4, (255, 0, 255), 2)  # top
    draw_line(p3, p6, (255, 0, 255), 2)  # bottom
    draw_line(p1, p3, (0, 255, 0), 2)  # left
    draw_line(p4, p6, (0, 255, 0), 2)  # right

    status = "PASSED" if m.passed else "FAILED"
    status_color = (0, 220, 0) if m.passed else (0, 0, 255)
    lines = [
        (status, status_color),
        (f"Top:    {m.top_w:.1f}px" if math.isfinite(m.top_w) else "Top:    -", (230, 230, 230)),
        (f"Mid:    {m.mid_w:.1f}px" if math.isfinite(m.mid_w) else "Mid:    -", (230, 230, 230)),
        (f"Bottom: {m.bottom_w:.1f}px" if math.isfinite(m.bottom_w) else "Bottom: -", (230, 230, 230)),
        (f"Diff: {m.width_diff_px:.1f}px ({m.width_diff_pct*100:.1f}%)" if math.isfinite(m.width_diff_px) else "Diff: -", (230, 230, 230)),
        (f"Ratio (B/T): {m.bottom_over_top:.3f}" if math.isfinite(m.bottom_over_top) else "Ratio (B/T): -", (230, 230, 230)),
        (f"Angle(v): {m.angle_center_v:.2f}°" if math.isfinite(m.angle_center_v) else "Angle(v): -", (230, 230, 230)),
        (f"L edge:  {m.angle_left_v:.2f}°" if math.isfinite(m.angle_left_v) else "L edge:  -", (230, 230, 230)),
        (f"R edge:  {m.angle_right_v:.2f}°" if math.isfinite(m.angle_right_v) else "R edge:  -", (230, 230, 230)),
    ]
    if m.missing:
        lines.append((f"Missing: {','.join(m.missing)}", (0, 165, 255)))

    out = draw_panel(frame_bgr, panel_w, lines)
    return out, m


def main():
    ap = argparse.ArgumentParser(
        description="Run YOLOv8 pose inference and immediately compute PASS/FAIL based on widths + straightness, overlaying results on images/video."
    )
    ap.add_argument("--model", type=Path, default=Path("runs/pose/train/weights/best.pt"), help="Path to best.pt")
    ap.add_argument("--source", type=str, required=True, help="Image, folder, video path, or 0 for webcam")
    ap.add_argument("--out", type=Path, default=Path("runs/pose/graded"), help="Output folder (images/video + report.csv)")
    ap.add_argument("--imgsz", type=int, default=1024, help="Inference image size (default 1024)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (default 0.25)")
    ap.add_argument("--kpt-min-conf", type=float, default=0.25, help="Min keypoint confidence (default 0.25)")
    ap.add_argument("--width-tol-px", type=float, default=10.0, help="Width tolerance in pixels (default 10). Use only if distance is fixed.")
    ap.add_argument("--width-tol-pct", type=float, default=0.08, help="Width tolerance ratio (default 0.08=8%). Recommended when distance varies.")
    ap.add_argument(
        "--width-mode",
        type=str,
        default="pct",
        choices=["pct", "px", "either"],
        help="How to evaluate top/bottom width equality (default: pct). Use pct when distance varies.",
    )
    ap.add_argument("--angle-tol-deg", type=float, default=5.0, help="Straightness tolerance in degrees (default 5)")
    ap.add_argument("--use-mid-width", action="store_true", help="Also enforce mid width consistency (p2-p5).")
    ap.add_argument("--panel-w", type=int, default=360, help="Side panel width (default 360)")
    args = ap.parse_args()

    from ultralytics import YOLO  # type: ignore
    import cv2  # type: ignore

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    # Decide if source is webcam/video vs images
    src = args.source
    is_cam = src.strip() == "0"
    src_path = Path(src) if not is_cam else None

    report_path = args.out / "report.csv"
    report_rows = []

    if is_cam or (src_path and src_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}):
        cap = cv2.VideoCapture(0 if is_cam else str(src_path))
        if not cap.isOpened():
            raise SystemExit("Failed to open video/webcam source")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_video = args.out / ("webcam_graded.mp4" if is_cam else f"{src_path.stem}_graded.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w + args.panel_w, h))

        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            graded, m = process_frame(
                frame,
                model=model,
                imgsz=args.imgsz,
                conf=args.conf,
                kpt_min_conf=args.kpt_min_conf,
                width_tol_px=args.width_tol_px,
                width_tol_pct=args.width_tol_pct,
                angle_tol_deg=args.angle_tol_deg,
                use_mid_width=args.use_mid_width,
                width_mode=args.width_mode,
                panel_w=args.panel_w,
            )
            writer.write(graded)

            if m is not None:
                report_rows.append(
                    {
                        "frame": frame_i,
                        "passed": int(m.passed),
                        "top_w_px": m.top_w,
                        "mid_w_px": m.mid_w,
                        "bottom_w_px": m.bottom_w,
                        "width_diff_px": m.width_diff_px,
                        "width_diff_pct": m.width_diff_pct,
                        "bottom_over_top": m.bottom_over_top,
                        "angle_center_v": m.angle_center_v,
                        "angle_left_v": m.angle_left_v,
                        "angle_right_v": m.angle_right_v,
                    }
                )
            frame_i += 1

        cap.release()
        writer.release()
        print(f"Wrote video: {out_video}")
    else:
        # images: single file or folder
        if src_path is None:
            raise SystemExit("Invalid image source")
        if src_path.is_dir():
            images = sorted([p for p in src_path.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".webp", ".bmp"}])
        else:
            images = [src_path]
        if not images:
            raise SystemExit("No images found")

        out_dir = args.out / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            graded, m = process_frame(
                frame,
                model=model,
                imgsz=args.imgsz,
                conf=args.conf,
                kpt_min_conf=args.kpt_min_conf,
                width_tol_px=args.width_tol_px,
                width_tol_pct=args.width_tol_pct,
                angle_tol_deg=args.angle_tol_deg,
                use_mid_width=args.use_mid_width,
                width_mode=args.width_mode,
                panel_w=args.panel_w,
            )
            cv2.imwrite(str(out_dir / img_path.name), graded)
            if m is not None:
                report_rows.append(
                    {
                        "image": img_path.name,
                        "passed": int(m.passed),
                        "top_w_px": m.top_w,
                        "mid_w_px": m.mid_w,
                        "bottom_w_px": m.bottom_w,
                        "width_diff_px": m.width_diff_px,
                        "width_diff_pct": m.width_diff_pct,
                        "bottom_over_top": m.bottom_over_top,
                        "angle_center_v": m.angle_center_v,
                        "angle_left_v": m.angle_left_v,
                        "angle_right_v": m.angle_right_v,
                    }
                )
        print(f"Wrote images to: {out_dir}")

    # Write CSV report
    if report_rows:
        with report_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(report_rows[0].keys())
            wcsv = csv.DictWriter(f, fieldnames=fieldnames)
            wcsv.writeheader()
            wcsv.writerows(report_rows)
        print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()


