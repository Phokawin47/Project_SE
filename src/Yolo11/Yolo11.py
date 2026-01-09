from ultralytics import YOLO
from pathlib import Path

def main():
    ROOT = Path(__file__).resolve().parent
    WEIGHT = ROOT / "weights" / "yolo11n.pt"
    DATA = ROOT / "dataset" / "data.yaml"

    # Load a model
    model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights


    model.train(
        data=str(DATA),
        epochs=300,
        imgsz=640,
        batch=8,
        patience=20,

        degrees=15,
        scale=0.2,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,

        project="runs",
        name="yolo11_exp"
    )

if __name__ == "__main__":
    main()
