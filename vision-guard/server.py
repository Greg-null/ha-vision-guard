"""Vision Guard – REST API für Objekterkennung und GIF-Erstellung."""

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# --- Konfiguration -----------------------------------------------------------

OPTIONS_PATH = "/data/options.json"
DEFAULT_OPTIONS = {
    "confidence_threshold": 0.45,
    "model_size": "yolov8n",
    "gif_fps": 8,
    "gif_width": 640,
    "log_level": "info",
}


def load_options():
    """Liest Addon-Optionen aus /data/options.json (HA Standard)."""
    if os.path.exists(OPTIONS_PATH):
        with open(OPTIONS_PATH) as f:
            return {**DEFAULT_OPTIONS, **json.load(f)}
    return DEFAULT_OPTIONS


options = load_options()

logging.basicConfig(
    level=getattr(logging, options["log_level"].upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vision-guard")

# --- YOLO Modell laden --------------------------------------------------------

MODEL_MAP = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
}

CLASS_LABELS_DE = {
    "person": "Person",
    "cat": "Katze",
    "dog": "Hund",
    "car": "Auto",
    "truck": "LKW",
    "bus": "Bus",
    "motorcycle": "Motorrad",
    "bicycle": "Fahrrad",
    "bird": "Vogel",
}

# YOLO Klassen-IDs für die relevanten Kategorien
PERSON_CLASSES = {0}  # person
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
ANIMAL_CLASSES = {14, 15, 16}  # bird, cat, dog

model_path = MODEL_MAP.get(options["model_size"], "yolov8n.pt")
log.info("Lade YOLO Modell: %s", model_path)
model = YOLO(model_path)
log.info("Modell geladen.")


# --- Hilfs-Funktionen ---------------------------------------------------------


def classify_detection(class_id):
    """Ordnet YOLO class_id einer Kategorie zu."""
    if class_id in PERSON_CLASSES:
        return "person"
    if class_id in VEHICLE_CLASSES:
        return "vehicle"
    if class_id in ANIMAL_CLASSES:
        return "animal"
    return "other"


def draw_detections(image_path, results, output_path):
    """Zeichnet Bounding Boxes mit Labels auf das Bild."""
    img = cv2.imread(image_path)
    if img is None:
        log.error("Bild konnte nicht geladen werden: %s", image_path)
        return False

    h, w = img.shape[:2]

    colors = {
        "person": (0, 0, 255),      # Rot
        "vehicle": (255, 165, 0),    # Orange
        "animal": (0, 255, 0),       # Grün
        "other": (128, 128, 128),    # Grau
    }

    for box in results:
        x1, y1, x2, y2 = map(int, box["bbox"])
        category = box["category"]
        color = colors.get(category, (128, 128, 128))
        label = f'{box["label"]} {box["confidence"]:.0%}'

        # Box zeichnen
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Label-Hintergrund
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, min(h, w) / 1200)
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), font, font_scale,
                    (255, 255, 255), thickness)

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


# --- API Endpoints ------------------------------------------------------------


@app.route("/detect", methods=["POST"])
def detect():
    """Objekterkennung auf einem Bild.

    POST JSON:
        image_path: Pfad zum Bild (im Container, z.B. /config/www/snapshots/...)
        hint: (optional) Was Reolink erkannt hat ("person", "pet", "vehicle")
        exclude_zones: (optional) Liste von Zonen die ignoriert werden sollen
            [{"x1": 0, "y1": 0, "x2": 100, "y2": 100}]

    Returns JSON:
        detected: bool – Wurde ein relevantes Objekt bestätigt?
        detections: Liste der Erkennungen mit bbox, label, confidence, category
        annotated_path: Pfad zum annotierten Bild
        inference_ms: Inferenz-Dauer in ms
    """
    data = request.get_json(force=True)
    image_path = data.get("image_path", "")
    hint = data.get("hint", "").lower()
    exclude_zones = data.get("exclude_zones", [])
    conf = data.get("confidence", options["confidence_threshold"])

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": f"Bild nicht gefunden: {image_path}"}), 404

    log.info("Erkennung gestartet: %s (Hinweis: %s)", image_path, hint or "keiner")
    t0 = time.time()

    # YOLO Inferenz
    results = model(image_path, conf=conf, verbose=False)
    inference_ms = (time.time() - t0) * 1000

    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            category = classify_detection(class_id)

            # Exclude-Zonen prüfen
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            in_exclude_zone = False
            for zone in exclude_zones:
                if (zone["x1"] <= cx <= zone["x2"] and
                        zone["y1"] <= cy <= zone["y2"]):
                    in_exclude_zone = True
                    break

            if in_exclude_zone:
                log.debug("Ausgeschlossen (Zone): %s bei (%.0f, %.0f)",
                          class_name, cx, cy)
                continue

            label_de = CLASS_LABELS_DE.get(class_name, class_name)
            detections.append({
                "label": label_de,
                "class_name": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 3),
                "bbox": [round(c) for c in bbox],
                "category": category,
            })

    # Relevante Erkennung = person, vehicle, oder animal
    relevant = [d for d in detections if d["category"] != "other"]

    # Wenn Reolink einen Hint gegeben hat, prüfe ob YOLO das bestätigt
    hint_map = {"person": "person", "pet": "animal", "vehicle": "vehicle"}
    hint_category = hint_map.get(hint, "")
    if hint_category:
        hint_confirmed = any(d["category"] == hint_category for d in relevant)
    else:
        hint_confirmed = len(relevant) > 0

    # Annotiertes Bild erstellen
    base, ext = os.path.splitext(image_path)
    annotated_path = f"{base}_annotated{ext}"
    if relevant:
        draw_detections(image_path, relevant, annotated_path)
    else:
        annotated_path = ""

    log.info("Erkennung fertig: %d Objekte, %d relevant, %.0fms",
             len(detections), len(relevant), inference_ms)

    return jsonify({
        "detected": hint_confirmed,
        "detections": detections,
        "relevant_count": len(relevant),
        "annotated_path": annotated_path,
        "inference_ms": round(inference_ms),
        "hint": hint,
        "hint_confirmed": hint_confirmed,
    })


@app.route("/gif", methods=["POST"])
def create_gif():
    """Konvertiert eine MP4-Datei in ein GIF.

    POST JSON:
        video_path: Pfad zur MP4-Datei
        output_path: (optional) Zielpfad für das GIF
        fps: (optional) Frames pro Sekunde (default: aus config)
        width: (optional) Breite in Pixel (default: aus config)

    Returns JSON:
        gif_path: Pfad zum erstellten GIF
        size_kb: Dateigröße in KB
        duration_ms: Konvertierungsdauer
    """
    data = request.get_json(force=True)
    video_path = data.get("video_path", "")
    fps = data.get("fps", options["gif_fps"])
    width = data.get("width", options["gif_width"])

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": f"Video nicht gefunden: {video_path}"}), 404

    # Output-Pfad bestimmen
    output_path = data.get("output_path", "")
    if not output_path:
        base = os.path.splitext(video_path)[0]
        output_path = f"{base}.gif"

    log.info("GIF-Erstellung: %s → %s", video_path, output_path)
    t0 = time.time()

    # ffmpeg: MP4 → GIF mit Palette für bessere Qualität
    palette_path = tempfile.mktemp(suffix=".png")
    try:
        # Schritt 1: Palette generieren
        cmd_palette = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen=stats_mode=diff",
            palette_path,
        ]
        subprocess.run(cmd_palette, capture_output=True, timeout=30, check=True)

        # Schritt 2: GIF mit Palette erstellen
        cmd_gif = [
            "ffmpeg", "-y", "-i", video_path, "-i", palette_path,
            "-lavfi", f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
            output_path,
        ]
        subprocess.run(cmd_gif, capture_output=True, timeout=60, check=True)
    except subprocess.CalledProcessError as e:
        log.error("ffmpeg Fehler: %s", e.stderr.decode() if e.stderr else str(e))
        return jsonify({"error": "GIF-Erstellung fehlgeschlagen",
                        "detail": e.stderr.decode() if e.stderr else str(e)}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "GIF-Erstellung Timeout"}), 504
    finally:
        if os.path.exists(palette_path):
            os.unlink(palette_path)

    duration_ms = (time.time() - t0) * 1000
    size_kb = os.path.getsize(output_path) / 1024 if os.path.exists(output_path) else 0

    log.info("GIF fertig: %.0f KB, %.0fms", size_kb, duration_ms)

    return jsonify({
        "gif_path": output_path,
        "size_kb": round(size_kb),
        "duration_ms": round(duration_ms),
    })


@app.route("/health", methods=["GET"])
def health():
    """Health-Check Endpoint."""
    return jsonify({
        "status": "ok",
        "model": options["model_size"],
        "confidence": options["confidence_threshold"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Vision Guard startet auf Port %d", port)
    app.run(host="0.0.0.0", port=port)
