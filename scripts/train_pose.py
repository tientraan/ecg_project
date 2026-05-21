from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

model.train(
    data="datasets/ecg_paper/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,
    project="runs",
    name="ecg_pose"
)