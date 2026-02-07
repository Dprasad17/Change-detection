#!/bin/bash
# =============================================================================
# EtherX Innovations - Production Deployment Script
# =============================================================================
# Usage: ./scripts/deploy.sh [environment]
# Example: ./scripts/deploy.sh production
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"
COMPOSE_FILE="docker-compose.yml"
COMPOSE_OVERRIDE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Validate environment
validate_environment() {
    log_info "Validating environment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check model file
    if [ ! -f "$PROJECT_DIR/saved_model.pth" ]; then
        log_warning "Model file (saved_model.pth) not found. AI inference will be disabled."
    fi
    
    # Check .env file
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_warning ".env file not found. Copying from .env.example..."
        if [ "$ENVIRONMENT" = "production" ] && [ -f "$PROJECT_DIR/.env.production" ]; then
            cp "$PROJECT_DIR/.env.production" "$PROJECT_DIR/.env"
        else
            cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        fi
    fi
    
    log_success "Environment validation passed"
}

# Set compose files based on environment
set_compose_files() {
    if [ "$ENVIRONMENT" = "production" ]; then
        if [ -f "$PROJECT_DIR/docker-compose.prod.yml" ]; then
            COMPOSE_OVERRIDE="-f docker-compose.prod.yml"
        fi
        PROFILE="--profile production"
    else
        PROFILE=""
    fi
    
    log_info "Using compose configuration: $COMPOSE_FILE $COMPOSE_OVERRIDE"
}

# Pull latest images
pull_images() {
    log_info "Pulling latest Docker images..."
    cd "$PROJECT_DIR"
    docker compose -f $COMPOSE_FILE $COMPOSE_OVERRIDE pull || true
    log_success "Images pulled successfully"
}

# Build application
build_application() {
    log_info "Building application..."
    cd "$PROJECT_DIR"
    docker compose -f $COMPOSE_FILE $COMPOSE_OVERRIDE build --no-cache
    log_success "Application built successfully"
}

# Stop existing services
stop_services() {
    log_info "Stopping existing services..."
    cd "$PROJECT_DIR"
    docker compose -f $COMPOSE_FILE $COMPOSE_OVERRIDE down --remove-orphans || true
    log_success "Services stopped"
}

# Start services
start_services() {
    log_info "Starting services..."
    cd "$PROJECT_DIR"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        docker compose -f $COMPOSE_FILE $COMPOSE_OVERRIDE $PROFILE up -d
    else
        docker compose -f $COMPOSE_FILE up -d
    fi
    
    log_success "Services started"
}

# Wait for health check
wait_for_health() {
    log_info "Waiting for application to be healthy..."
    
    MAX_RETRIES=30
    RETRY_COUNT=0
    HEALTH_URL="http://localhost:8006/health"
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
            log_success "Application is healthy!"
            return 0
        fi
        
        RETRY_COUNT=$((RETRY_COUNT + 1))
        log_info "Waiting for application... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 5
    done
    
    log_error "Application health check failed after $MAX_RETRIES attempts"
}

# Show status
show_status() {
    log_info "Current service status:"
    cd "$PROJECT_DIR"
    docker compose -f $COMPOSE_FILE $COMPOSE_OVERRIDE ps
    
    echo ""
    log_info "Application endpoints:"
    echo "  - Web UI: http://localhost:8006"
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "  - Web UI (via nginx): http://localhost"
    fi
    echo "  - Health: http://localhost:8006/health"
    echo "  - API Status: http://localhost:8006/api/status"
    echo "  - Metrics: http://localhost:8006/metrics"
}

# Cleanup old images
cleanup() {
    log_info "Cleaning up old Docker images..."
    docker image prune -f
    log_success "Cleanup completed"
}

# Main deployment flow
main() {
    echo "=============================================="
    echo "  EtherX Satellite Detection Deployment"
    echo "  Environment: $ENVIRONMENT"
    echo "=============================================="
    echo ""
    
    validate_environment
    set_compose_files
    
    # Uncomment to pull base images
    # pull_images
    
    build_application
    stop_services
    start_services
    wait_for_health
    show_status
    cleanup
    
    echo ""
    log_success "Deployment completed successfully!"
}

# Run main function
main
