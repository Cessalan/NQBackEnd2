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

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application using exec form (JSON array) for proper signal handling
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]