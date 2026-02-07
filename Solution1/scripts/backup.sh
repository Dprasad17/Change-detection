#!/bin/bash
# =============================================================================
# EtherX Innovations - Backup Script
# =============================================================================
# Usage: ./scripts/backup.sh [backup_dir]
# Example: ./scripts/backup.sh /backup/etherx
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_BASE="${1:-/backup/etherx}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_BASE/$TIMESTAMP"

# Colors
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

# Create backup directory
create_backup_dir() {
    log_info "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
}

# Backup configuration files
backup_config() {
    log_info "Backing up configuration files..."
    
    mkdir -p "$BACKUP_DIR/config"
    
    # Copy configuration files
    [ -f "$PROJECT_DIR/.env" ] && cp "$PROJECT_DIR/.env" "$BACKUP_DIR/config/"
    [ -f "$PROJECT_DIR/.env.production" ] && cp "$PROJECT_DIR/.env.production" "$BACKUP_DIR/config/"
    [ -f "$PROJECT_DIR/docker-compose.yml" ] && cp "$PROJECT_DIR/docker-compose.yml" "$BACKUP_DIR/config/"
    [ -f "$PROJECT_DIR/docker-compose.prod.yml" ] && cp "$PROJECT_DIR/docker-compose.prod.yml" "$BACKUP_DIR/config/"
    [ -f "$PROJECT_DIR/nginx.conf" ] && cp "$PROJECT_DIR/nginx.conf" "$BACKUP_DIR/config/"
    
    log_success "Configuration backed up"
}

# Backup output data
backup_data() {
    log_info "Backing up output data..."
    
    if [ -d "$PROJECT_DIR/data" ]; then
        tar -czf "$BACKUP_DIR/data.tar.gz" -C "$PROJECT_DIR" data/
        log_success "Data backed up"
    else
        log_warning "No data directory found, skipping"
    fi
}

# Backup uploads (optional - can be large)
backup_uploads() {
    if [ "${BACKUP_UPLOADS:-false}" = "true" ]; then
        log_info "Backing up uploads..."
        
        if [ -d "$PROJECT_DIR/uploads" ]; then
            tar -czf "$BACKUP_DIR/uploads.tar.gz" -C "$PROJECT_DIR" uploads/
            log_success "Uploads backed up"
        else
            log_warning "No uploads directory found, skipping"
        fi
    else
        log_info "Skipping uploads backup (set BACKUP_UPLOADS=true to include)"
    fi
}

# Backup model (optional - large file)
backup_model() {
    if [ "${BACKUP_MODEL:-false}" = "true" ]; then
        log_info "Backing up model file..."
        
        if [ -f "$PROJECT_DIR/saved_model.pth" ]; then
            cp "$PROJECT_DIR/saved_model.pth" "$BACKUP_DIR/"
            log_success "Model backed up"
        else
            log_warning "No model file found, skipping"
        fi
    else
        log_info "Skipping model backup (set BACKUP_MODEL=true to include)"
    fi
}

# Create backup manifest
create_manifest() {
    log_info "Creating backup manifest..."
    
    cat > "$BACKUP_DIR/manifest.txt" << EOF
EtherX Backup Manifest
======================
Timestamp: $TIMESTAMP
Source: $PROJECT_DIR
Backup Location: $BACKUP_DIR

Contents:
EOF
    
    ls -la "$BACKUP_DIR" >> "$BACKUP_DIR/manifest.txt"
    
    log_success "Manifest created"
}

# Cleanup old backups (keep last N)
cleanup_old_backups() {
    KEEP_BACKUPS="${KEEP_BACKUPS:-7}"
    
    log_info "Cleaning up old backups (keeping last $KEEP_BACKUPS)..."
    
    cd "$BACKUP_BASE"
    ls -dt */ 2>/dev/null | tail -n +$((KEEP_BACKUPS + 1)) | xargs -r rm -rf
    
    log_success "Cleanup completed"
}

# Show backup summary
show_summary() {
    echo ""
    echo "=============================================="
    echo "  Backup Completed Successfully"
    echo "=============================================="
    echo "  Location: $BACKUP_DIR"
    echo "  Size: $(du -sh "$BACKUP_DIR" | cut -f1)"
    echo ""
    echo "  Contents:"
    ls -la "$BACKUP_DIR"
    echo "=============================================="
}

# Main
main() {
    echo "=============================================="
    echo "  EtherX Satellite Detection Backup"
    echo "=============================================="
    echo ""
    
    create_backup_dir
    backup_config
    backup_data
    backup_uploads
    backup_model
    create_manifest
    cleanup_old_backups
    show_summary
}

main
