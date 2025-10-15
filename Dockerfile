FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System packages needed for faiss
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first and install
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app code
COPY . .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# ✅ FIXED: Use shell form to allow environment variable substitution
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
