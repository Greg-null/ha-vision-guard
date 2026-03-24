ARG BUILD_FROM=python:3.12-slim
FROM ${BUILD_FROM}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n model on build
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY server.py .
COPY run.sh /run.sh
RUN chmod a+x /run.sh

EXPOSE 5000

ENTRYPOINT ["/run.sh"]
