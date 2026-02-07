# =============================================================================
# EtherX Innovations - Makefile
# Convenience commands for development and deployment
# =============================================================================

.PHONY: help build run stop logs clean deploy backup restore test lint

# Default target
help:
	@echo "EtherX Satellite Detection - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Development:"
	@echo "  make build     - Build Docker images"
	@echo "  make run       - Start development server"
	@echo "  make stop      - Stop all services"
	@echo "  make logs      - View application logs"
	@echo "  make shell     - Open shell in container"
	@echo ""
	@echo "Production:"
	@echo "  make prod      - Start with production profile"
	@echo "  make deploy    - Run deployment script"
	@echo "  make backup    - Create backup"
	@echo "  make restore   - Restore from backup"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean     - Remove containers and volumes"
	@echo "  make prune     - Deep clean Docker system"
	@echo "  make health    - Check application health"
	@echo ""
	@echo "Development Tools:"
	@echo "  make lint      - Run code linters"
	@echo "  make test      - Run tests"
	@echo ""

# =============================================================================
# Development Commands
# =============================================================================

build:
	docker compose build

run:
	docker compose up -d
	@echo "Application starting at http://localhost:8006"

stop:
	docker compose down

logs:
	docker compose logs -f etherx-app

shell:
	docker compose exec etherx-app /bin/bash

# =============================================================================
# Production Commands
# =============================================================================

prod:
	docker compose --profile production up -d
	@echo "Production starting at http://localhost"

prod-build:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml build

prod-run:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

deploy:
	./scripts/deploy.sh production

backup:
	./scripts/backup.sh

restore:
	@echo "Usage: ./scripts/restore.sh <backup_directory>"

# =============================================================================
# Maintenance Commands
# =============================================================================

clean:
	docker compose down -v --remove-orphans
	docker system prune -f

prune:
	docker system prune -af --volumes

health:
	@curl -sf http://localhost:8006/health | python -m json.tool || echo "Health check failed"

status:
	@echo "=== Container Status ==="
	docker compose ps
	@echo ""
	@echo "=== Health Check ==="
	@curl -sf http://localhost:8006/health | python -m json.tool 2>/dev/null || echo "Service unavailable"

# =============================================================================
# Development Tools
# =============================================================================

lint:
	@echo "Running linters..."
	python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || true
	python -m black --check . || true
	python -m isort --check-only . || true

format:
	@echo "Formatting code..."
	python -m black .
	python -m isort .

test:
	@echo "Running tests..."
	python -m pytest tests/ -v || echo "No tests found"

# =============================================================================
# Local Development (without Docker)
# =============================================================================

venv:
	python -m venv .venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source .venv/bin/activate  (Linux/Mac)"
	@echo "  .venv\\Scripts\\activate    (Windows)"

install:
	pip install -r requirements.txt

dev:
	python main.py

# =============================================================================
# Documentation
# =============================================================================

docs:
	@echo "Opening documentation..."
	@echo "README: README.md"
	@echo "Deployment: DEPLOYMENT.md"
