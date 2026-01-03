import io
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st


# Local imports from your repo
from predict_and_grade import Metrics, process_frame  # noqa: E402


APP_TITLE = "Shuttering Evaluation Demo"
DEFAULT_MODEL_PATHS = [
    Path("runs/pose/train/weights/best.pt"),
    Path("runs/pose/train/weights/last.pt"),
    Path("yolov8n-pose.pt"),
]


def _css():
    st.markdown(
        """
<style>
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
  .title-wrap { display: flex; gap: 14px; align-items: baseline; }
  .subtitle { color: rgba(250,250,250,0.75); font-size: 0.95rem; }
  .card {
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 14px 16px;
  }
  .metric-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 700;
    letter-spacing: 0.2px;
  }
  .ok { background: rgba(0, 200, 90, 0.18); border: 1px solid rgba(0, 200, 90, 0.45); }
  .bad { background: rgba(255, 60, 60, 0.16); border: 1px solid rgba(255, 60, 60, 0.45); }
  .warn { background: rgba(255, 190, 0, 0.16); border: 1px solid rgba(255, 190, 0, 0.45); }
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import YOLO  # imported lazily so Streamlit loads fast

    return YOLO(model_path)


def find_default_model() -> Optional[Path]:
    for p in DEFAULT_MODEL_PATHS:
        if p.exists():
            return p
    return None


def bgr_from_uploaded_image(data: bytes):
    import cv2  # type: ignore

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def write_bytes_to_temp(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    return path


def reencode_h264_mp4(in_path: str, out_path: str) -> bool:
    """
    Re-encode to a browser-friendly MP4 (H.264, yuv420p, faststart).
    Returns True if successful.
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            in_path,
            "-an",  # no audio
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path,
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return p.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def _badge_html(label: str, kind: str) -> str:
    cls = {"ok": "badge ok", "bad": "badge bad", "warn": "badge warn"}[kind]
    return f'<span class="{cls}">{label}</span>'


def render_metrics(m: Optional[Metrics], width_mode: str):
    if m is None:
        st.markdown(_badge_html("NO DETECTION", "warn"), unsafe_allow_html=True)
        st.caption("No pose detected in this frame. Try adjusting confidence thresholds or ensure the subject is visible.")
        return

    st.markdown(
        _badge_html("PASSED" if m.passed else "FAILED", "ok" if m.passed else "bad"),
        unsafe_allow_html=True,
    )

    with st.container(border=False):
        st.markdown('<div class="card">', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Top width (px)", f"{m.top_w:.1f}" if np.isfinite(m.top_w) else "-")
        c2.metric("Bottom width (px)", f"{m.bottom_w:.1f}" if np.isfinite(m.bottom_w) else "-")
        c3.metric("B/T ratio", f"{m.bottom_over_top:.3f}" if np.isfinite(m.bottom_over_top) else "-")

        c4, c5, c6 = st.columns(3)
        c4.metric("Diff (px)", f"{m.width_diff_px:.1f}" if np.isfinite(m.width_diff_px) else "-")
        c5.metric("Diff (%)", f"{m.width_diff_pct*100:.1f}%" if np.isfinite(m.width_diff_pct) else "-")
        c6.metric("Angle (deg)", f"{m.angle_center_v:.2f}" if np.isfinite(m.angle_center_v) else "-")

        st.caption(
            f"Width mode: **{width_mode}** — for variable distance, use **pct** (scale-invariant)."
        )
        if m.missing:
            st.warning("Missing/low-confidence keypoints: " + ", ".join(m.missing))

        st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📐", layout="wide")
    _css()

    # Header
    st.markdown(
        """
<div class="title-wrap">
  <h2 style="margin:0;">Shuttering Evaluation Demo</h2>
  <div class="subtitle">Upload an image or video → run pose inference → get PASS/FAIL with widths + straightness.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.subheader("Model")
        default_model = find_default_model()
        model_path_text = st.text_input(
            "Model path",
            value=str(default_model) if default_model else "",
            help="Path to your trained YOLOv8 pose weights (e.g., runs/pose/train/weights/best.pt).",
        )
        model_upload = st.file_uploader("…or upload a .pt model", type=["pt"])
        st.divider()

        st.subheader("Scoring")
        width_mode = st.selectbox(
            "Width mode (recommended: pct for variable distance)",
            options=["pct", "px", "either"],
            index=0,
        )
        width_tol_px = st.slider("Width tolerance (px)", 0, 50, 10, 1)
        width_tol_pct = st.slider("Width tolerance (%)", 0.0, 30.0, 8.0, 0.5) / 100.0
        angle_tol_deg = st.slider("Straightness tolerance (deg)", 0.5, 20.0, 5.0, 0.5)
        use_mid_width = st.toggle("Also enforce mid-width consistency (p2↔p5)", value=True)
        st.divider()

        st.subheader("Inference")
        imgsz = st.select_slider("Image size", options=[640, 768, 896, 1024, 1280], value=1024)
        conf = st.slider("Detection confidence", 0.01, 0.9, 0.25, 0.01)
        kpt_min_conf = st.slider("Keypoint min confidence", 0.01, 0.9, 0.25, 0.01)
        panel_w = st.select_slider("Overlay panel width", options=[280, 320, 360, 420, 480], value=360)

        st.caption("Tip: If you miss detections, raise `imgsz` or lower confidence thresholds slightly.")

    # Resolve model path (uploaded model wins)
    model_path: Optional[str] = None
    tmp_model_path: Optional[str] = None
    if model_upload is not None:
        tmp_model_path = write_bytes_to_temp(model_upload.getvalue(), suffix=".pt")
        model_path = tmp_model_path
    elif model_path_text.strip():
        model_path = model_path_text.strip()

    if not model_path:
        st.info("Set a model path in the sidebar (e.g. `runs/pose/train/weights/best.pt`) or upload a `.pt` model.")
        return

    # Main content
    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        st.subheader("Upload")
        mode = st.radio("Input type", options=["Image", "Video"], horizontal=True)
        if mode == "Image":
            up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
            if up is None:
                st.caption("Upload an image to run the demo.")
                return

            img_bgr = bgr_from_uploaded_image(up.getvalue())
            if img_bgr is None:
                st.error("Could not read image.")
                return

            with st.spinner("Loading model…"):
                model = load_model(model_path)

            with st.spinner("Running inference + grading…"):
                graded_bgr, metrics = process_frame(
                    img_bgr,
                    model=model,
                    imgsz=imgsz,
                    conf=conf,
                    kpt_min_conf=kpt_min_conf,
                    width_tol_px=width_tol_px,
                    width_tol_pct=width_tol_pct,
                    angle_tol_deg=angle_tol_deg,
                    use_mid_width=use_mid_width,
                    width_mode=width_mode,
                    panel_w=panel_w,
                )

            # Show result
            st.image(graded_bgr[:, :, ::-1], caption="Graded output", use_container_width=True)

            # Download
            import cv2  # type: ignore

            ok, buf = cv2.imencode(".jpg", graded_bgr)
            if ok:
                st.download_button(
                    "Download graded image",
                    data=buf.tobytes(),
                    file_name="graded.jpg",
                    mime="image/jpeg",
                )

            with right:
                st.subheader("Result")
                render_metrics(metrics, width_mode)
                if metrics is not None:
                    with st.expander("Raw metrics (debug)"):
                        st.json(asdict(metrics))

        else:
            up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
            if up is None:
                st.caption("Upload a video to run the demo.")
                return

            # Save to temp
            tmp_in = write_bytes_to_temp(up.getvalue(), suffix=Path(up.name).suffix or ".mp4")
            tmp_out_raw = tempfile.mktemp(suffix=".mp4")
            tmp_out_h264 = tempfile.mktemp(suffix=".mp4")

            with st.spinner("Loading model…"):
                model = load_model(model_path)

            st.info("Processing video… this runs per-frame and can take time depending on length and CPU/GPU.")
            progress = st.progress(0.0)
            status = st.empty()

            import cv2  # type: ignore

            cap = cv2.VideoCapture(tmp_in)
            if not cap.isOpened():
                st.error("Failed to open uploaded video.")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_out_raw, fourcc, fps, (w + panel_w, h))

            report_rows = []
            frame_i = 0
            passed = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                graded, m = process_frame(
                    frame,
                    model=model,
                    imgsz=imgsz,
                    conf=conf,
                    kpt_min_conf=kpt_min_conf,
                    width_tol_px=width_tol_px,
                    width_tol_pct=width_tol_pct,
                    angle_tol_deg=angle_tol_deg,
                    use_mid_width=use_mid_width,
                    width_mode=width_mode,
                    panel_w=panel_w,
                )
                writer.write(graded)

                if m is not None and m.passed:
                    passed += 1

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
                if total_frames > 0:
                    progress.progress(min(frame_i / total_frames, 1.0))
                else:
                    progress.progress(min((frame_i % 200) / 200.0, 1.0))
                status.write(f"Frames processed: {frame_i}" + (f" / {total_frames}" if total_frames else ""))

            cap.release()
            writer.release()
            progress.progress(1.0)
            status.write(f"Done. Frames: {frame_i} | Passed frames: {passed}")

            # Re-encode to browser-friendly MP4 (fixes black/non-playable video in many browsers)
            ok_h264 = reencode_h264_mp4(tmp_out_raw, tmp_out_h264)
            playable_path = tmp_out_h264 if ok_h264 else tmp_out_raw

            # Show video + downloads
            st.subheader("Graded video")
            st.video(playable_path)

            with open(playable_path, "rb") as f:
                st.download_button(
                    "Download graded video",
                    data=f.read(),
                    file_name="graded_video.mp4",
                    mime="video/mp4",
                )

            if report_rows:
                # create CSV in-memory
                import pandas as pd  # type: ignore

                df = pd.DataFrame(report_rows)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download report.csv",
                    data=csv_bytes,
                    file_name="report.csv",
                    mime="text/csv",
                )

            with right:
                st.subheader("Summary")
                if frame_i:
                    st.metric("Frames processed", frame_i)
                    st.metric("Passed frames", passed)
                    st.metric("Pass rate", f"{(passed / frame_i) * 100:.1f}%")
                st.caption("Tip: increase width tolerance (%) if your camera viewpoint has strong perspective.")

    # Cleanup temp model if uploaded
    if tmp_model_path:
        try:
            os.remove(tmp_model_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()


