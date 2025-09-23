FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# copy and install python requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r requirements.txt

# copy app
COPY . /app

# environment hint for YOLO cache dir (optional)
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# default command; Render will override with --port $PORT when needed
CMD ["python", "-m", "uvicorn", "scanner_server_deepsort:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
