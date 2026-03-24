# Vision Guard

Lokale Objekterkennung mit YOLOv8 und GIF-Erstellung für Home Assistant Kamera-Überwachung.

## Features

- **Objekterkennung** mit YOLOv8 (Person, Tier, Fahrzeug)
- **Bounding Boxes** werden direkt ins Bild gezeichnet
- **GIF-Erstellung** aus MP4-Aufnahmen mit optimierter Palette
- **Exclude-Zonen** zum Ausschließen bekannter Objekte (z.B. geparktes Auto)
- **CPU-optimiert** für Intel N95 und ähnliche Prozessoren

## API Endpoints

### POST /detect

Analysiert ein Bild und gibt Erkennungen zurück.

```json
{
  "image_path": "/config/www/snapshots/parkplatz.jpg",
  "hint": "person",
  "exclude_zones": [{"x1": 100, "y1": 200, "x2": 400, "y2": 500}],
  "confidence": 0.45
}
```

### POST /gif

Konvertiert ein MP4-Video in ein GIF.

```json
{
  "video_path": "/config/www/snapshots/parkplatz.mp4",
  "fps": 8,
  "width": 640
}
```

### GET /health

Gibt den Status des Addons zurück.

## Konfiguration

| Option | Standard | Beschreibung |
|--------|----------|-------------|
| confidence_threshold | 0.45 | Mindest-Konfidenz für Erkennung |
| model_size | yolov8n | Modellgröße (n=schnell, s=mittel, m=genau) |
| gif_fps | 8 | Frames pro Sekunde im GIF |
| gif_width | 640 | GIF-Breite in Pixel |
| log_level | info | Log-Level |
