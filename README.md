# Shuttering_Evaluation

YOLOv8 Pose training + inference utilities for evaluating whether a person fits a rectangular "box" and is standing straight.

## What’s included
- `labelme_to_yolo_pose.py`: convert LabelMe JSON (`Ready/`) → YOLOv8-pose dataset (`dataset/`)
- `predict_and_grade.py`: **one-step** inference + PASS/FAIL overlay for images or video
- `geometry_check_pose.py`: compute widths/angles from `save_txt` outputs → CSV
- `annotate_geometry.py`: overlay PASS/FAIL on images using `save_txt` outputs
- `xanylabeling_pose_config.yaml`: keypoint order + skeleton for X-AnyLabeling

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## One-step inference + grading (recommended)

Images/folder:

```powershell
.\venv\Scripts\Activate.ps1
python .\predict_and_grade.py --source .\Ready --out .\runs\pose\graded --width-mode pct --width-tol-pct 0.08 --angle-tol-deg 5 --use-mid-width
```

Video:

```powershell
.\venv\Scripts\Activate.ps1
python .\predict_and_grade.py --source \"your_video.mp4\" --out .\runs\pose\graded_video --width-mode pct --width-tol-pct 0.08 --angle-tol-deg 5 --use-mid-width
```

Outputs go under the `--out` directory (annotated images/video + `report.csv`).


