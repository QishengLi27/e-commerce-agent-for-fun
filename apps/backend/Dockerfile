FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and install production dependencies
COPY pyproject.toml .
COPY backend/ ./backend/
COPY data/ ./data/
RUN pip install --no-cache-dir -e .

# Expose FastAPI port
EXPOSE 8000

# Run with Gunicorn + Uvicorn workers for production concurrency
CMD ["gunicorn", "backend.api.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
