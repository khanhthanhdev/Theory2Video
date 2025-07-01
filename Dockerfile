# Optimized Dockerfile for Hugging Face Spaces
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables for HF Spaces
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    PATH="/root/.TinyTeX/bin/x86_64-linux:$PATH" \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1

# Install system dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    gcc \
    g++ \
    build-essential \
    pkg-config \
    portaudio19-dev \
    libasound2-dev \
    libsdl-pango-dev \
    libcairo2-dev \
    libpango1.0-dev \
    sox \
    ffmpeg \
    texlive-full \
    dvisvgm \
    ghostscript \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir  -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -c "import gradio; print(f'Gradio version: {gradio.__version__}')" \
    && find /usr/local -name "*.pyc" -delete \
    && find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Ensure Hugging Face cache directories are writable

# Create models directory and download models efficiently
RUN mkdir -p models && cd models \
    && echo "Downloading models for HF Spaces..." \
    && wget --progress=dot:giga --timeout=30 --tries=3 \
        -O kokoro-v0_19.onnx \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx" \
    && wget --progress=dot:giga --timeout=30 --tries=3 \
        -O voices.bin \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin" \
    && ls -la /app/models/

# Copy all project files and folders
COPY . .

ENV PYTHONPATH="/app:$PYTHONPATH"
RUN echo "export PYTHONPATH=/app:\$PYTHONPATH" >> ~/.bashrc

# Run embedding creation script at build time
RUN python create_embeddings.py

# Ensure all files are writable (fix PermissionError for log file)
RUN chmod -R a+w /app

# Create output directory
RUN mkdir -p output tmp
# Ensure output and tmp directories are writable (fix PermissionError for session_id.txt)
RUN chmod -R a+w /app/output /app/tmp || true

RUN mkdir -p output tmp logs \
    && mkdir -p /app/.cache/huggingface/hub \
    && mkdir -p /app/.cache/transformers \
    && mkdir -p /app/.cache/sentence_transformers \
    && chmod -R 755 /app/.cache \
    && chmod 755 /app/models \
    && ls -la /app/models/ \
    && echo "Cache directories created with proper permissions"
# Add HF Spaces specific metadata
LABEL space.title="Text 2 Mnaim" \
      space.sdk="docker" \
      space.author="khanhthanhdev" \
      space.description="Text to science video using multi Agent"

# Expose the port that HF Spaces expects
EXPOSE 7860

# Health check optimized for HF Spaces
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=2 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the Gradio app with HF Spaces optimized settings
CMD ["python", "gradio_app.py"]