FROM python:3.11-slim

# Install system dependencies for OCR, image processing, and document handling
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libpoppler-cpp-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Old Office files (.ppt/.doc/.xls) are converted to .pptx/.docx/.xlsx before
# reading (core/file_format.py convert_legacy); no Python library reads them.
# Measured 2026-10-14 on the production image: +148 MB compressed, ~2s per
# file, ~230 MB RAM per conversion, no change to container start-up.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-impress \
    libreoffice-writer \
    libreoffice-calc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure WebSocket support is installed regardless of requirements.txt encoding issues
RUN pip install --no-cache-dir "uvicorn[standard]==0.34.0" websockets

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application using exec form (JSON array) for proper signal handling
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]