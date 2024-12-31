# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory and download/extract the model
RUN mkdir -p /app/model
RUN wget -O /app/model/moondream-2b-int8.mf.gz \
    https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz && \
    cd /app/model && \
    gzip -d moondream-2b-int8.mf.gz && \
    mv moondream-2b-int8.mf moondream-2b-int8.bin

# Copy application code
COPY app.py .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]