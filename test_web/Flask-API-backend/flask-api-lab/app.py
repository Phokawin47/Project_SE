from flask import Flask, jsonify, request
from ultralytics import YOLO
import base64
import cv2
import io
from PIL import Image
from flask_cors import CORS  # ← import CORS

app = Flask(__name__)

# ⭐ เพิ่ม CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://10.54.55.59:3000","http://10.199.22.24:3000"],  # อนุญาตแค่ Frontend
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

model = YOLO("yolo11m")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        results = model.predict(image)
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        res_image = Image.fromarray(res_rgb)
        buffered = io.BytesIO()
        res_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        detections = []
        if results[0].boxes is None or len(results[0].boxes) == 0:
            detections.append({
                "class": "ไม่พบวัตถุ",
                "conf": None
            })
        else:
            for cls_id, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
                detections.append({
                    "class": results[0].names[int(cls_id)],
                    "conf": float(conf)
                })
        return jsonify({
            "detections": detections,
            "imagedetect": img_str
        })
    else:
        return jsonify({
            "Error Text": "ไฟล์ผิดประเภท"
        })

@app.route("/", methods=["GET"])
def mainRoute():
    return "Hello Flask API"

@app.route("/health", methods=["GET"])
def check_health():
    return jsonify({
        "Status": 200
    })
    
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "ไม่รู้จัก Route ที่เรียกใช้ครับ",
        "code": 404
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error": "Method ไม่ถูกต้องครับ",
        "code": 405
    }), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "error": "Internal Server Error",
        "code": 500
    }), 500

@app.errorhandler(502)
def bad_gateway(e):
    return jsonify({
        "error": "Bad Gateway",
        "code": 502
    }), 502

@app.errorhandler(503)
def service_unavailable(e):
    return jsonify({
        "error": "Service Unavailable",
        "code": 503
    }), 503

@app.errorhandler(504)
def gateway_timeout(e):
    return jsonify({
        "error": "Gateway Timeout",
        "code": 504
    }), 504

@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        "error": "ส่งข้อมูลมาไม่ถูก pattern",
        "code": 400
    }), 400
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2569, debug=True)