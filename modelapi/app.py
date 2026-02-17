from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

import base64
import io
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000","http://10.225.0.12:3000","10.225.100.190:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ✅ SocketIO (เลือก cors_allowed_origins ให้ตรง frontend)
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000","http://10.225.0.12:3000","10.225.100.190:3000"])

# ✅ โหลดโมเดล best.pt
MODEL_PATH = Path(__file__).resolve().parent / "best11.pt"
model = YOLO(str(MODEL_PATH))

# warmup (ช่วยให้ request แรกไม่หน่วง)
try:
    model.predict(Image.new("RGB", (240, 240)), verbose=False)
except Exception:
    pass


@app.route("/", methods=["GET"])
def mainRoute():
    return "Hello Flask SocketIO API"

@app.route("/health", methods=["GET"])
def check_health():
    return jsonify({"Status": 200})


def infer_base64_jpeg_to_result(b64_jpeg: str):
    """รับ base64(jpeg) -> return {detections, imagedetect} เหมือนเดิม"""
    img_bytes = base64.b64decode(b64_jpeg)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    results = model.predict(image, verbose=False)

    # วาดรูป output
    plotted = results[0].plot()
    res_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    res_image = Image.fromarray(res_rgb)

    buff = io.BytesIO()
    res_image.save(buff, format="JPEG", quality=85)
    out_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    detections = []
    if results[0].boxes is None or len(results[0].boxes) == 0:
        detections.append({"class": "ไม่พบวัตถุ", "conf": None})
    else:
        for cls_id, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
            detections.append({
                "class": results[0].names[int(cls_id)],
                "conf": float(conf)
            })

    return {"detections": detections, "imagedetect": out_b64}


@socketio.on("frame")
def on_frame(data):
    """
    data = { image: "<base64 jpeg>" }
    """
    try:
        b64 = data.get("image")
        if not b64:
            emit("pred", {"error": "missing image"})
            return

        payload = infer_base64_jpeg_to_result(b64)
        emit("pred", payload)

    except Exception as e:
        emit("pred", {"error": str(e)})


if __name__ == "__main__":
    # ✅ สำคัญ: ใช้ socketio.run ไม่ใช่ app.run
    socketio.run(app, host="0.0.0.0", port=2569, debug=True)
