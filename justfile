# Justfile for RAG Chatbot development commands
# Run `just` to see all available commands

# Default: show all available recipes
default:
    @just --list

# Format code with Black
format:
    uv run black backend/ main.py

# Check code formatting without making changes
check-format:
    uv run black --check backend/ main.py

# Run tests
test:
    cd backend && uv run pytest -v

# Run tests with coverage report
test-cov:
    cd backend && uv run pytest --cov=. --cov-report=term-missing --cov-report=html

# Run all quality checks (format + tests)
lint: check-format test

# Start the development server
run:
    ./run.sh

# Install/sync dependencies
sync:
    uv sync --group dev

# Add a new dependency
add-dep package:
    uv add {{package}}

# Add a new dev dependency
add-dev-dep package:
    uv add --group dev {{package}}
