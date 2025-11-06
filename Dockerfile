# Dockerfile.api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first for reliability on slim images
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.0

# App deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py .

# Health check env
ENV PORT=8000
EXPOSE 8000

# Start
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
