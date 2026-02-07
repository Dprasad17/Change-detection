FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    HOST=0.0.0.0 \
    PORT=7860

# Create non-root user
RUN groupadd --gid 1000 user && \
    useradd --uid 1000 --gid user --shell /bin/bash --create-home user

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    gdal-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY Solution1/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code from subdirectory
COPY --chown=user:user Solution1/ .

# Create necessary directories
RUN mkdir -p uploads data/web_output && \
    chown -R user:user /app

# Switch to non-root user
USER user

# Expose Hugging Face port
EXPOSE 7860

# Run commands
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
