"""
Trash Detection Web App
------------------------
Flask backend that serves a YOLOv8 object-detection model (trained on the
Trash-Net dataset: Cardboard, Metal, Plastic) for image uploads via a browser.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000

Deploy:
    See README.md for Render / Railway / Docker instructions.
"""

import os
import io
import uuid
import logging

from flask import Flask, request, render_template, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit
CONF_THRESHOLD = 0.5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trash-detector")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# --------------------------------------------------------------------------
# Load model once at startup (never per-request -- keeps things fast & avoids
# repeated disk/model-init errors under load)
# --------------------------------------------------------------------------
model = None
model_load_error = None

def load_model():
    global model, model_load_error
    try:
        from ultralytics import YOLO
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model weights not found at '{MODEL_PATH}'. "
                f"Place your trained 'best.pt' inside the 'model/' folder."
            )
        model = YOLO(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        model_load_error = str(e)
        logger.error("Failed to load model: %s", model_load_error)

load_model()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None, model_error=model_load_error)


@app.route("/health")
def health():
    """Simple health-check endpoint for uptime monitors / hosting platforms."""
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({"status": status, "error": model_load_error}), (200 if model else 503)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Model is not loaded: {model_load_error}"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use form field name 'file'."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    try:
        results = model.predict(source=image, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": model.names[cls_id],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(v), 2) for v in box.xyxy[0].tolist()],
            })

        # Save annotated image so the frontend can display it
        annotated = result.plot()  # numpy array, BGR
        annotated_img = Image.fromarray(annotated[..., ::-1])  # BGR -> RGB

        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        annotated_img.save(save_path, quality=90)

        return jsonify({
            "detections": detections,
            "count": len(detections),
            "annotated_image_url": url_for("static", filename=f"results/{filename}"),
        })

    except Exception as e:  # noqa: BLE001
        logger.exception("Inference failed")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File too large. Max upload size is 10 MB."}), 413


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(500)
def server_error(_e):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
