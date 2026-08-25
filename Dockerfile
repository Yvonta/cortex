# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

# Install system dependencies, ffmpeg, git, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Upgrade pip and install NumPy 1.26.0
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy==1.26.0"

# Copy dependency definition and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Chatterbox directly from GitHub
RUN pip install --no-cache-dir --upgrade git+https://github.com/resemble-ai/chatterbox.git

# Copy application source code and entrypoint
COPY . .

# Expose port 80 for FastAPI and port 11434 for Ollama
EXPOSE 80 11434

ENTRYPOINT ["/app/entrypoint.sh"]