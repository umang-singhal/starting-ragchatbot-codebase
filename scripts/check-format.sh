#!/bin/bash
# Check code formatting with Black (without making changes)

set -e

echo "Checking code formatting with Black..."
uv run black --check backend/ main.py "$@"
echo "✨ Code formatting check passed!"
