# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster AS builder

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install dependencies with more robust installation
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --default-timeout=200 \
    --retries 5 \
    -r requirements.txt

# Final stage
FROM python:3.9-slim-buster

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY Rulebase/ ./Rulebase/

# Set PYTHONPATH to include the current directory and Rulebase
ENV PYTHONPATH=/app:/app/Rulebase

# Install additional web framework dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    python-multipart

# Expose a range of ports
EXPOSE 2000-2999

# Define environment variable for port
ENV PORT=2222

# Define environment variable
ENV NAME=ExerciseAnalysisAPI

# Run the application with port from environment variable
CMD ["sh", "-c", "uvicorn Rulebase.exercise_api:app --host 0.0.0.0 --port ${PORT} --proxy-headers"] 