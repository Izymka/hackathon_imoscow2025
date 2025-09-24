# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Prevents Python from writing .pyc files and ensures stdout/stderr are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps (optional, add if SimpleITK or others require)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY chest_ct_ai_classifier/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy project
COPY . /app

EXPOSE 8000

# Run the FastAPI service
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
