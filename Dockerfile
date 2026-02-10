# RunPod Serverless SDXL + LoRA Worker
# Optimized for 24GB VRAM GPUs (L40S, A5000, 3090, etc.)

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY rp_handler.py .

# Environment variables for model
# Override MODEL_ID in RunPod console to use different checkpoints
ENV MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start handler
CMD ["python", "-u", "rp_handler.py"]
