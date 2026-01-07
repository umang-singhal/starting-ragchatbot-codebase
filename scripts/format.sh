#!/bin/bash
# Format Python code with Black

set -e

echo "Formatting Python code with Black..."
uv run black backend/ main.py "$@"
echo "✨ Code formatted successfully!"
