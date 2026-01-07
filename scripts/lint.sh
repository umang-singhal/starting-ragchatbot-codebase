#!/bin/bash
# Run all code quality checks

set -e

echo "Running code quality checks..."
echo ""

echo "1. Checking code formatting with Black..."
uv run black --check backend/ main.py

echo ""
echo "2. Running tests with pytest..."
cd backend && uv run pytest -v "$@"

echo ""
echo "✨ All quality checks passed!"
