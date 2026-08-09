# Trash Detector — Web Deployment

A Flask web app that serves your YOLOv8 trash-detection model (Cardboard,
Metal, Plastic) for image uploads through a browser.

## 1. Get your trained model

Your `Untitled1.ipynb` notebook trains the model on Colab and saves it to:

```
/content/drive/MyDrive/YOLOv8_Trash_Model/trash_detection_best.pt
```

Download that file from Google Drive, rename it to **`best.pt`**, and place
it in this project's `model/` folder:

```
deploy/
  model/
    best.pt   <-- put it here
```

The app will refuse to start inference (with a clear error message, not a
crash) if this file is missing, so you'll always know why detection isn't
working.

## 2. Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**, upload an image, and view detections.

Check `http://127.0.0.1:5000/health` any time to confirm the model loaded.

## 3. Project structure

```
deploy/
  app.py                # Flask backend + inference logic
  requirements.txt       # Pinned, tested dependencies
  Dockerfile             # Production container (gunicorn, not the dev server)
  model/
    best.pt              # Your trained weights (you add this)
  static/
    style.css
    results/              # Annotated output images are saved here
  templates/
    index.html            # Upload UI
```

## 4. Deploy to a live website

### Option A — Render.com (easiest, free tier available)
1. Push this folder to a GitHub repo (weights included, or pulled at build
   time — see note below on file size).
2. On Render: **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn --bind 0.0.0.0:$PORT app:app`
5. Render sets `$PORT` automatically — `app.py` already reads it.

### Option B — Railway.app
1. `railway init` in this folder, `railway up`.
2. Railway auto-detects Python and uses `requirements.txt`.
3. Set the start command to `gunicorn --bind 0.0.0.0:$PORT app:app` in
   Railway's settings.

### Option C — Docker (any host: Fly.io, AWS, GCP, a VPS, etc.)
```bash
docker build -t trash-detector .
docker run -p 5000:5000 trash-detector
```

### A note on `best.pt` file size
YOLOv8n weights are small (~6 MB), so committing `model/best.pt` directly to
git is fine. If you later train a larger model (s/m/l/x) and it exceeds
GitHub's 100 MB limit, use [Git LFS](https://git-lfs.com/) or download the
weights at container startup from cloud storage (S3, Google Drive direct
link, Hugging Face Hub, etc.) instead of committing the binary.

## 5. Common errors this setup already prevents

| Symptom | Cause it avoids |
|---|---|
| `libGL.so.1: cannot open shared object file` | Uses `opencv-python-headless`, not `opencv-python` |
| App crashes on startup if weights are missing | `app.py` catches the load error and shows it on `/` and `/health` instead of crashing |
| 500 error on huge uploads | `MAX_CONTENT_LENGTH` = 10 MB, returns a clean 413 JSON error |
| Non-image file uploaded | Extension + PIL decode validation before inference |
| "Works on my machine" dependency drift | All versions in `requirements.txt` are pinned and were installed + run end-to-end in a clean environment |
| Dev server used in production (insecure/slow) | `Dockerfile` runs `gunicorn`, not `flask run` / `python app.py` |

## 6. API reference

`POST /predict` — multipart form field `file` (image). Returns:
```json
{
  "count": 2,
  "detections": [
    {"class": "Plastic", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}
  ],
  "annotated_image_url": "/static/results/<uuid>.jpg"
}
```

`GET /health` — `{"status": "ok"}` (200) or `{"status": "model_not_loaded", "error": "..."}` (503).
