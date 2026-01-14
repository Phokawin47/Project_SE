from ultralytics import YOLO
from pathlib import Path
import torch
import ultralytics


# ====== แก้ 2 บรรทัดนี้ให้ตรงเครื่องคุณ ======
DATA_DIR = Path(r"C:/Users/HP/Downloads/yolo/archive/Natural Hand Digits Dataset_split")  # โฟลเดอร์ที่มี train/valid(or val)
PROJECT_DIR = Path(r"C:/Users/HP\Downloads/yolo/runs")          # ที่เก็บผลลัพธ์/weights
# ==============================================

RUN_NAME = "finger_cls_yolo12"

def find_yolo12_cls_model() -> str:
    """
    พยายามหาโมเดล YOLOv12 สำหรับ classification:
    1) ถ้ามีไฟล์ weight .pt อยู่ในโฟลเดอร์นี้ -> ใช้เลย
    2) ถ้าไม่มี -> หาไฟล์ yolo12*-cls.yaml ในแพ็กเกจ ultralytics แล้วใช้ path เต็ม
    """
    # 1) local weights (ถ้าคุณดาวน์โหลดไว้เอง)
    for w in ["yolo12n-cls.pt", "yolo12s-cls.pt", "yolo12m-cls.pt", "yolo12l-cls.pt", "yolo12x-cls.pt"]:
        if Path(w).exists():
            return w

    # 2) search yaml inside ultralytics package
    pkg_root = Path(ultralytics.__file__).resolve().parent  # .../site-packages/ultralytics
    candidates = ["yolo12n-cls.yaml", "yolo12-cls.yaml", "yolo12s-cls.yaml", "yolo12m-cls.yaml", "yolo12l-cls.yaml", "yolo12x-cls.yaml"]

    for name in candidates:
        found = list(pkg_root.rglob(name))
        if found:
            return str(found[0])

    # ถ้าหา YOLOv12 ไม่เจอจริง ๆ
    raise FileNotFoundError(
        "หาไฟล์ YOLOv12 classification ไม่เจอ (ทั้ง .pt และ .yaml)\n"
        "ลองอัปเดต ultralytics: python -m pip install -U ultralytics\n"
        "หรือบอกผมว่าในโฟลเดอร์ ultralytics/cfg มีไฟล์อะไรบ้าง เดี๋ยวผมชี้ไฟล์ที่ถูกให้"
    )

def main():
    # ตั้ง device อัตโนมัติ: ถ้ามี CUDA ใช้ GPU0 ไม่งั้นใช้ CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # (แนะนำ) ถ้าโฟลเดอร์คุณชื่อ valid ให้ใช้ได้ แต่ถ้าเจอปัญหาให้ rename เป็น val
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

    model_path = find_yolo12_cls_model()
    print("Using model:", model_path)

    model = YOLO(model_path)

    model.train(
        data=str(DATA_DIR),
        epochs=20,
        imgsz=224,
        batch=8,                # CPU แนะนำ 4-8; ถ้า GPU ค่อยเพิ่ม
        device="cpu",
        workers=2,

        # ===== Save/Resume ที่สำคัญ =====
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        save=True,
        save_period=1,          # เซฟทุก 1 epoch กันพังสุด
        resume=False            # ครั้งแรกให้เป็น False
    )

if __name__ == "__main__":
    main()
