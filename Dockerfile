FROM python:3.11-slim

LABEL maintainer="AdaptNXT Technology Solutions <queries@adaptnxt.com>"
LABEL description="Edge AI Computer Vision & Jetson Zone Sentinel Pipeline"
LABEL org.opencontainers.image.source="https://github.com/adaptnxt/jetson-yolo-rtsp-pipeline"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["adaptnxt-vision"]
CMD ["demo"]
