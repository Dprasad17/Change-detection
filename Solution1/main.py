"""
EtherX Innovations - Satellite Change Detection
Production-Ready FastAPI Application

AI-powered satellite imagery change detection using Siamese UNet architecture.
"""

import os
import shutil
import uuid
import secrets
import json
from pathlib import Path
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
import pyproj

os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import torch
import numpy as np
import rasterio
from PIL import Image

# Import core logic from Dataset_Testing
from Dataset_Testing import (
    UNet, load_trained_model, process_test_pair,
    run_inference_on_test_data,
    INFERENCE_PARAMS, PATCH_SIZE, PATCH_OVERLAP, DEVICE
)

# =============================================================================
# CONFIGURATION
# =============================================================================
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8006,http://127.0.0.1:8006").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
API_KEY = os.getenv("API_KEY")  # Optional API key authentication

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
class JSONFormatter(logging.Formatter):
    """JSON log formatter for production environments."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if hasattr(record, 'correlation_id'):
            log_obj["correlation_id"] = record.correlation_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logging():
    """Configure logging based on environment."""
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()

    if LOG_FORMAT == "json" or ENVIRONMENT == "production":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    console_handler.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler("web_server.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(log_level)

    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger("EtherX")


logger = setup_logging()

# =============================================================================
# SECURITY MIDDLEWARE
# =============================================================================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        if ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            # Content Security Policy for production
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://unpkg.com; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://unpkg.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: blob:; "
                "connect-src 'self'"
            )

        return response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Add correlation ID to requests for tracing."""

    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id

        return response


# =============================================================================
# RATE LIMITING
# =============================================================================
limiter = Limiter(key_func=get_remote_address)

# =============================================================================
# API KEY AUTHENTICATION (Optional)
# =============================================================================
async def verify_api_key(request: Request):
    """Verify API key if configured."""
    if API_KEY is None:
        return True

    api_key = request.headers.get("X-API-Key")
    if not api_key or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return True


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================
model_instance = None
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "saved_model.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global model_instance

    # Startup
    logger.info(f"🚀 Starting EtherX Innovations Server")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Rate Limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds")

    if MODEL_PATH.exists():
        logger.info(f"Loading model from {MODEL_PATH}")
        try:
            model_instance = load_trained_model(str(MODEL_PATH))
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
    else:
        logger.warning(f"⚠️ Model file not found at {MODEL_PATH}")

    yield

    # Shutdown
    logger.info("👋 Shutting down EtherX server")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================
app = FastAPI(
    title="EtherX Innovations - Satellite Change Detection",
    description="AI-powered satellite imagery change detection using Siamese UNet architecture",
    version="1.0.0",
    docs_url="/api/docs" if DEBUG else None,
    redoc_url="/api/redoc" if DEBUG else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Paths
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "data" / "web_output"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
ALLOWED_EXTENSIONS = {'.zip', '.tif', '.tiff'}
ALLOWED_MIME_TYPES = {'application/zip', 'image/tiff', 'application/octet-stream'}


def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file."""
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False

    # Check content type (basic check)
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        # Allow octet-stream as fallback
        if file.content_type != 'application/octet-stream':
            return False

    return True


def sanitize_job_id(job_id: str) -> str:
    """Sanitize job ID to prevent path traversal."""
    # Only allow alphanumeric and hyphens
    sanitized = ''.join(c for c in job_id if c.isalnum() or c == '-')
    return sanitized[:36]  # UUID length


def convert_tif_to_png(tif_path, png_path, is_mask=False):
    """Utility to convert a 1-band mask or 3-band image to a displayable PNG."""
    try:
        with rasterio.open(tif_path) as src:
            data = src.read()
            bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]

            if is_mask:
                img = data[0]
                rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
                mask = img > 0
                rgba[mask] = [255, 20, 147, 200]
                Image.fromarray(rgba).save(png_path)
            else:
                img = data[:3].transpose(1, 2, 0)
                img = img.astype(np.float32)
                for i in range(3):
                    band = img[:, :, i]
                    p2, p98 = np.percentile(band, (2, 98))
                    img[:, :, i] = np.clip((band - p2) / (p98 - p2 + 1e-5) * 255, 0, 255)
                img = img.astype(np.uint8)
                Image.fromarray(img).save(png_path)

            return bounds
    except Exception as e:
        logger.error(f"Failed to convert {tif_path} to PNG: {e}")
        return None


def generate_qgis_project(job_id, t1_path, t2_path, mask_path):
    """Generates a QGIS .qgs XML project file."""
    qgs_content = f"""<qgis projectname="EtherX Analysis - {job_id}" version="3.28.4-Firenze">
  <homePath path=""/>
  <projectlayers>
    <maplayer autoRefreshEnabled="0" autoRefreshTime="0" hasScaleBasedVisibilityFlag="0" maxScale="0" minScale="1e+08" refreshOnNotifyEnabled="0" refreshOnNotifyMessage="" styleCategories="AllStyleCategories" type="raster">
      <id>T1_Before_{job_id}</id>
      <datasource>{t1_path}</datasource>
      <layername>T1: Before Image</layername>
    </maplayer>
    <maplayer autoRefreshEnabled="0" autoRefreshTime="0" hasScaleBasedVisibilityFlag="0" maxScale="0" minScale="1e+08" refreshOnNotifyEnabled="0" refreshOnNotifyMessage="" styleCategories="AllStyleCategories" type="raster">
      <id>T2_After_{job_id}</id>
      <datasource>{t2_path}</datasource>
      <layername>T2: After Image</layername>
    </maplayer>
    <maplayer autoRefreshEnabled="0" autoRefreshTime="0" hasScaleBasedVisibilityFlag="0" maxScale="0" minScale="1e+08" refreshOnNotifyEnabled="0" refreshOnNotifyMessage="" styleCategories="AllStyleCategories" type="raster">
      <id>Change_Mask_{job_id}</id>
      <datasource>{mask_path}</datasource>
      <layername>AI Change Mask</layername>
      <renderer-v2 alpha="0.7" nodataColor="" opacity="1" type="singlebandcolordata">
        <rasterrenderer alphaBand="-1" band="1" opacity="1" type="singlebandcolordata"/>
      </renderer-v2>
    </maplayer>
  </projectlayers>
  <layerorder>
    <layer id="Change_Mask_{job_id}"/>
    <layer id="T2_After_{job_id}"/>
    <layer id="T1_Before_{job_id}"/>
  </layerorder>
</qgis>"""
    project_path = OUTPUT_DIR / job_id / "project.qgs"
    with open(project_path, "w", encoding="utf-8") as f:
        f.write(qgs_content)
    return project_path


# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================
@app.get("/health", tags=["System"])
@limiter.limit(f"{RATE_LIMIT_REQUESTS * 10}/minute")
async def health_check(request: Request):
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "model_loaded": model_instance is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/status", tags=["System"])
@limiter.limit(f"{RATE_LIMIT_REQUESTS}/minute")
async def api_status(request: Request):
    """Detailed system status for monitoring dashboards."""
    return {
        "service": "EtherX Satellite Analysis",
        "version": "1.0.0",
        "model_ready": model_instance is not None,
        "device": str(DEVICE),
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "environment": ENVIRONMENT
    }


@app.get("/metrics", tags=["System"])
@limiter.limit("60/minute")
async def metrics(request: Request):
    """Prometheus-compatible metrics endpoint."""
    # Basic metrics
    metrics_data = f"""# HELP etherx_up Server up status
# TYPE etherx_up gauge
etherx_up 1

# HELP etherx_model_loaded Model loading status
# TYPE etherx_model_loaded gauge
etherx_model_loaded {1 if model_instance is not None else 0}
"""
    return HTMLResponse(content=metrics_data, media_type="text/plain")


# =============================================================================
# MAIN ROUTES
# =============================================================================
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def index(request: Request):
    """Serve the main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", tags=["Analysis"])
@limiter.limit(f"{RATE_LIMIT_REQUESTS}/minute")
async def upload_pair(
    request: Request,
    t1: UploadFile = File(...),
    t2: UploadFile = File(...)
):
    """
    Upload satellite image pair for change detection analysis.

    - **t1**: Before image (T1) - .zip or .tif format
    - **t2**: After image (T2) - .zip or .tif format
    """
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')

    # Check if model is loaded
    if model_instance is None:
        logger.error(f"[{correlation_id}] Model not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI Model not initialized. Please check server logs."
        )

    # Validate files
    if not validate_file(t1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid T1 file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    if not validate_file(t2):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid T2 file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)

    logger.info(f"[{correlation_id}] Starting job {job_id}")

    try:
        # Save uploaded files
        t1_ext = Path(t1.filename).suffix.lower()
        t2_ext = Path(t2.filename).suffix.lower()

        t1_save_path = job_dir / f"T1{t1_ext}"
        t2_save_path = job_dir / f"T2{t2_ext}"

        # Read and check file size
        t1_content = await t1.read()
        if len(t1_content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"T1 file exceeds maximum size of {MAX_FILE_SIZE_MB}MB"
            )

        t2_content = await t2.read()
        if len(t2_content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"T2 file exceeds maximum size of {MAX_FILE_SIZE_MB}MB"
            )

        # Write files
        with open(t1_save_path, "wb") as buffer:
            buffer.write(t1_content)
        with open(t2_save_path, "wb") as buffer:
            buffer.write(t2_content)

        # Create pair zip for processing
        import zipfile

        def ensure_zip(input_path, output_name):
            zip_path = job_dir / f"{output_name}.zip"
            with zipfile.ZipFile(zip_path, 'w') as z:
                z.write(input_path, arcname=input_path.name)
            return zip_path

        t1_zip = ensure_zip(t1_save_path, "T1_wrapped")
        t2_zip = ensure_zip(t2_save_path, "T2_wrapped")

        pair_zip_path = job_dir / "Pair1.zip"
        with zipfile.ZipFile(pair_zip_path, 'w') as zipf:
            zipf.write(t1_zip, "T1_raw.zip")
            zipf.write(t2_zip, "T2_raw.zip")

        # Process
        results_root = OUTPUT_DIR / job_id
        prep_dir = results_root / "preprocess"
        out_dir = results_root / "output"
        prep_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)

        # Preprocess
        logger.info(f"[{correlation_id}] Preprocessing job {job_id}")
        processed = process_test_pair(pair_zip_path, 1, str(prep_dir))

        if not processed:
            logger.error(f"[{correlation_id}] Preprocessing failed for job {job_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Preprocessing failed. Ensure files contain valid satellite bands (Blue, Green, Red, NIR)."
            )

        # Inference
        logger.info(f"[{correlation_id}] Running inference for job {job_id}")
        success = run_inference_on_test_data(str(prep_dir), str(MODEL_PATH), str(out_dir))

        if not success:
            logger.error(f"[{correlation_id}] Inference failed for job {job_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI inference failed. Please try again or contact support."
            )

        # Post-process
        mask_tif = out_dir / "Test_Predicted_Change_1.tif"
        t1_tif = prep_dir / "T1" / "old_image_1.tif"
        t2_tif = prep_dir / "T2" / "new_image_1.tif"

        logger.info(f"[{correlation_id}] Converting results to PNG for job {job_id}")
        mask_bounds = convert_tif_to_png(mask_tif, out_dir / "mask.png", is_mask=True)
        t1_bounds = convert_tif_to_png(t1_tif, prep_dir / "t1_display.png")
        t2_bounds = convert_tif_to_png(t2_tif, prep_dir / "t2_display.png")

        # Generate QGIS Project
        generate_qgis_project(job_id, str(t1_tif), str(t2_tif), str(mask_tif))

        logger.info(f"[{correlation_id}] Job {job_id} completed successfully")

        return {
            "job_id": job_id,
            "mask_url": f"/results/{job_id}/output/mask.png",
            "t1_url": f"/results/{job_id}/preprocess/t1_display.png",
            "t2_url": f"/results/{job_id}/preprocess/t2_display.png",
            "download_url": f"/results/{job_id}/output/Test_Predicted_Change_1.tif",
            "qgis_url": f"/results/{job_id}/project.qgs",
            "bounds": t1_bounds or t2_bounds
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{correlation_id}] Critical error in job {job_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again."
        )


@app.get("/results/{job_id}/{filename}", tags=["Results"])
async def get_project(job_id: str, filename: str):
    """Download project file."""
    job_id = sanitize_job_id(job_id)
    file_path = OUTPUT_DIR / job_id / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Project file not found")


@app.get("/results/{job_id}/{folder}/{filename}", tags=["Results"])
async def get_result(job_id: str, folder: str, filename: str):
    """Download result file."""
    job_id = sanitize_job_id(job_id)
    # Sanitize folder name
    folder = ''.join(c for c in folder if c.isalnum() or c in '-_')
    file_path = OUTPUT_DIR / job_id / folder / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Result file not found")


# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for production."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(f"[{correlation_id}] Unhandled exception: {exc}")

    if DEBUG:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "correlation_id": correlation_id}
        )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again.",
            "correlation_id": correlation_id
        }
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8006"))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
