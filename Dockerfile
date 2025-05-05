FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including LibreOffice
RUN apt-get update && apt-get install -y \
    libreoffice \
    libgl1-mesa-glx \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directories for file storage
RUN mkdir -p uploads converted

# Expose port
EXPOSE 8001

# Set environment variable to indicate server environment
ENV RUNNING_IN_CONTAINER=1

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"] 