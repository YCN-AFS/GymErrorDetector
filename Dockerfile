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
    websockets

# Make port 2222 available to the world outside this container
EXPOSE 2222

# Define environment variable
ENV NAME=ExerciseAnalysisAPI

# Run the application
CMD ["uvicorn", "Rulebase.exercise_api:app", "--host", "0.0.0.0", "--port", "2222", "--proxy-headers"] 