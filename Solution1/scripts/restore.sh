#!/bin/bash
# =============================================================================
# EtherX Innovations - Restore Script
# =============================================================================
# Usage: ./scripts/restore.sh [backup_dir]
# Example: ./scripts/restore.sh /backup/etherx/20260204_120000
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${1}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

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

# Validate backup directory
validate_backup() {
    if [ -z "$BACKUP_DIR" ]; then
        log_error "Usage: ./restore.sh <backup_directory>"
    fi
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_error "Backup directory not found: $BACKUP_DIR"
    fi
    
    if [ ! -f "$BACKUP_DIR/manifest.txt" ]; then
        log_warning "No manifest found in backup directory"
    fi
    
    log_success "Backup directory validated"
}

# Stop services before restore
stop_services() {
    log_info "Stopping services..."
    cd "$PROJECT_DIR"
    docker compose down || true
    log_success "Services stopped"
}

# Restore configuration
restore_config() {
    log_info "Restoring configuration..."
    
    if [ -d "$BACKUP_DIR/config" ]; then
        cp "$BACKUP_DIR/config/"* "$PROJECT_DIR/" 2>/dev/null || true
        log_success "Configuration restored"
    else
        log_warning "No configuration backup found"
    fi
}

# Restore data
restore_data() {
    log_info "Restoring data..."
    
    if [ -f "$BACKUP_DIR/data.tar.gz" ]; then
        tar -xzf "$BACKUP_DIR/data.tar.gz" -C "$PROJECT_DIR"
        log_success "Data restored"
    else
        log_warning "No data backup found"
    fi
}

# Restore uploads (optional)
restore_uploads() {
    if [ -f "$BACKUP_DIR/uploads.tar.gz" ]; then
        log_info "Restoring uploads..."
        tar -xzf "$BACKUP_DIR/uploads.tar.gz" -C "$PROJECT_DIR"
        log_success "Uploads restored"
    fi
}

# Restore model (optional)
restore_model() {
    if [ -f "$BACKUP_DIR/saved_model.pth" ]; then
        log_info "Restoring model..."
        cp "$BACKUP_DIR/saved_model.pth" "$PROJECT_DIR/"
        log_success "Model restored"
    fi
}

# Restart services
start_services() {
    log_info "Starting services..."
    cd "$PROJECT_DIR"
    
    if [ -f "docker-compose.prod.yml" ]; then
        docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    else
        docker compose up -d
    fi
    
    log_success "Services started"
}

# Main
main() {
    echo "=============================================="
    echo "  EtherX Satellite Detection Restore"
    echo "=============================================="
    echo "  Restoring from: $BACKUP_DIR"
    echo "  Target: $PROJECT_DIR"
    echo "=============================================="
    echo ""
    
    read -p "This will overwrite existing data. Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restore cancelled"
        exit 0
    fi
    
    validate_backup
    stop_services
    restore_config
    restore_data
    restore_uploads
    restore_model
    start_services
    
    echo ""
    log_success "Restore completed successfully!"
}

main
