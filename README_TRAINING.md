## What I set up for you

Your `Ready/` folder is LabelMe JSON with:
- one rectangle labeled `box`
- six keypoints labeled `point1..point6` in this meaning/order:
  - point1 = top-left
  - point2 = mid-left
  - point3 = bottom-left
  - point4 = top-right
  - point5 = mid-right
  - point6 = bottom-right

This repo now includes:
- `labelme_to_yolo_pose.py`: converts `Ready/` → `dataset/` in Ultralytics YOLOv8-pose format
- `xanylabeling_pose_config.yaml`: a pose label config you can load into X-AnyLabeling

## 1) Convert LabelMe → YOLO Pose dataset

From `C:\Users\abc\Desktop\ComputerVisionProjects\Training`:

```powershell
python .\labelme_to_yolo_pose.py --ready .\Ready --out .\dataset --clean
```

Outputs:
- `dataset/images/train`, `dataset/images/val`
- `dataset/labels/train`, `dataset/labels/val`
- `dataset/dataset.yaml` (includes `flip_idx: [3, 4, 5, 0, 1, 2]`)
- `dataset/conversion_summary.json` (skipped files, missing keypoints, etc.)

## 2) Train (Ultralytics YOLOv8 Pose)

Install:

```powershell
pip install ultralytics
```

Train:

```powershell
yolo pose train data=dataset/dataset.yaml model=yolov8n-pose.pt imgsz=1024 epochs=200 batch=8
```

Predict:

```powershell
yolo pose predict model=runs/pose/train/weights/best.pt source=dataset/images/val
```

## 3) X-AnyLabeling export config

Load `xanylabeling_pose_config.yaml` inside X-AnyLabeling (project/labels config), then export to YOLO Pose.


