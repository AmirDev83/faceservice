FROM python:3.11-slim

# System deps untuk OpenCV & InsightFace (ONNX Runtime CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps dulu (layer terpisah agar cache hit saat code berubah)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# InsightFace model (buffalo_l ~500MB) akan di-download ke INSIGHTFACE_HOME saat pertama run
# Volume faceservice_models di-mount ke sini supaya tidak re-download tiap restart
ENV INSIGHTFACE_HOME=/app/.insightface

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1"]
