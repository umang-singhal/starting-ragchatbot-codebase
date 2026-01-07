#!/bin/bash
# Run tests with coverage report

set -e

echo "Running tests with coverage..."
cd backend && uv run pytest --cov=. --cov-report=term-missing --cov-report=html "$@"

echo ""
echo "Coverage report generated in htmlcov/index.html"
