#!/bin/bash
# Run tests with pytest

set -e

echo "Running tests with pytest..."
cd backend && uv run pytest -v "$@"
