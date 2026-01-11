# =========================================================
# Customer Intelligence AI - Dockerfile
# =========================================================
#
# Build:   docker build -t customer-ai .
# Run:     docker run -p 8000:8000 -e OPENAI_API_KEY=your_key customer-ai
#
# =========================================================

# Use official Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Set the app environment
ENV APP_ENV=production
ENV LOG_LEVEL=INFO

# Install system dependencies
# - gcc: Required for some Python packages
# - libffi-dev: Foreign function interface for Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Create directories for runtime data
RUN mkdir -p /app/data/uploads /app/data/vector_store && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
# Using 4 workers for production (adjust based on CPU cores)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
