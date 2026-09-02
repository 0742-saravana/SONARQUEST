# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies required by OpenCV and Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create the model_weights directory
RUN mkdir -p backend/model_weights

# IMPORTANT: Download the model from your GitHub Release
# REPLACE THE URL BELOW with the actual raw download URL of your best.pt file from your NEW GitHub Releases
RUN wget -qO backend/model_weights/best.pt "https://github.com/0742-saravana/SONARQUEST/releases/download/V2.0/best.pt"

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run the FastAPI server using the PORT environment variable provided by Cloud Run
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
