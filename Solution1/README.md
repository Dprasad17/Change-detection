# EtherX Innovations - Satellite Change Detection

<div align="center">

![EtherX Logo](static/img/etherx_logo.jpg)

**AI-Powered Satellite Imagery Change Detection**

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

</div>

---

## 🌍 Overview

EtherX Satellite Change Detection is an enterprise-grade AI platform for detecting changes in multi-temporal satellite imagery using a state-of-the-art Siamese UNet deep learning architecture. Upload before/after satellite images and receive accurate change detection masks with full GIS integration.

### Key Features

- **🧠 AI-Powered Analysis**: Siamese UNet architecture for sub-pixel change detection
- **🗺️ GIS Integration**: Full QGIS project export with georeferenced outputs
- **🚀 Production Ready**: Docker deployment with nginx, rate limiting, and monitoring
- **🔒 Secure**: Security headers, input validation, and optional API key authentication
- **📊 Scalable**: Gunicorn workers with async processing

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU (recommended for faster inference)
- Model file: `saved_model.pth`

### Development Mode

```bash
# Clone the repository
git clone https://github.com/etherx-innovations/satellite-detection.git
cd satellite-detection

# Copy environment configuration
cp .env.example .env

# Start the application
docker-compose up --build

# Access at http://localhost:8006
```

### Production Mode

```bash
# Use production configuration
cp .env.production .env

# Start with nginx reverse proxy
docker-compose --profile production up -d

# Or use production overrides
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Access at http://localhost (port 80)
```

---

## 📁 Project Structure

```
etherx-satellite-detection/
├── main.py                 # FastAPI application
├── Dataset_Testing.py      # AI inference logic
├── Dataset_Training.py     # Model training (reference)
├── saved_model.pth         # Trained model weights
├── requirements.txt        # Python dependencies
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker orchestration
├── docker-compose.prod.yml # Production overrides
├── nginx.conf              # Nginx reverse proxy config
├── .env.example            # Environment template
├── .env.production         # Production environment
├── static/                 # Frontend assets
│   ├── css/style.css
│   ├── js/main.js
│   └── img/
├── templates/
│   └── index.html          # Main UI template
├── scripts/                # Deployment scripts
└── .github/workflows/      # CI/CD pipelines
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment mode (development/staging/production) | `development` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8006` |
| `MAX_FILE_SIZE_MB` | Maximum upload file size | `500` |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated) | `http://localhost:8006` |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | `INFO` |
| `LOG_FORMAT` | Log format (text/json) | `text` |
| `RATE_LIMIT_REQUESTS` | Max requests per window | `100` |
| `RATE_LIMIT_WINDOW` | Rate limit window in seconds | `60` |
| `API_KEY` | Optional API key for authentication | _(none)_ |
| `WORKERS` | Gunicorn worker count | `4` |

---

## 🔌 API Reference

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "environment": "production",
  "model_loaded": true,
  "timestamp": "2026-02-04T12:00:00.000000"
}
```

### System Status

```bash
GET /api/status
```

Response:
```json
{
  "service": "EtherX Satellite Analysis",
  "version": "1.0.0",
  "model_ready": true,
  "device": "cuda",
  "max_file_size_mb": 500,
  "environment": "production"
}
```

### Upload & Analyze

```bash
POST /upload
Content-Type: multipart/form-data

Parameters:
- t1: File (T1 before image - .zip or .tif)
- t2: File (T2 after image - .zip or .tif)
```

Response:
```json
{
  "job_id": "uuid",
  "mask_url": "/results/{job_id}/output/mask.png",
  "t1_url": "/results/{job_id}/preprocess/t1_display.png",
  "t2_url": "/results/{job_id}/preprocess/t2_display.png",
  "download_url": "/results/{job_id}/output/Test_Predicted_Change_1.tif",
  "qgis_url": "/results/{job_id}/project.qgs",
  "bounds": [[lat1, lon1], [lat2, lon2]]
}
```

### Metrics (Prometheus)

```bash
GET /metrics
```

---

## 🔒 Security

### Features

- **Security Headers**: HSTS, CSP, X-Frame-Options, X-Content-Type-Options
- **Rate Limiting**: Configurable per-IP rate limits
- **Input Validation**: File type and size validation
- **Path Sanitization**: Protection against path traversal attacks
- **Non-root Container**: Application runs as unprivileged user
- **Correlation IDs**: Request tracing for debugging

### API Key Authentication (Optional)

To enable API key authentication:

```bash
# In .env
API_KEY=your-secure-api-key-here
```

Include in requests:
```bash
curl -H "X-API-Key: your-secure-api-key-here" http://localhost:8006/upload ...
```

---

## 🐳 Docker Commands

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f etherx-app

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Check container health
docker-compose ps

# Enter container shell
docker-compose exec etherx-app /bin/bash
```

---

## 📊 Monitoring

### Prometheus Metrics

The `/metrics` endpoint exposes Prometheus-compatible metrics:

```
# HELP etherx_up Server up status
# TYPE etherx_up gauge
etherx_up 1

# HELP etherx_model_loaded Model loading status
# TYPE etherx_model_loaded gauge
etherx_model_loaded 1
```

### Health Checks

Docker health checks run every 30 seconds:
```bash
curl -f http://localhost:8006/health
```

---

## 🛠️ Development

### Local Setup (Without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run development server
python main.py
```

### API Documentation

In development mode, access interactive API docs:
- Swagger UI: http://localhost:8006/api/docs
- ReDoc: http://localhost:8006/api/redoc

---

## 📝 License

Copyright © 2026 EtherX Innovations. All rights reserved.

---

## 📞 Support

For enterprise support and licensing inquiries, contact: support@etherx.io
