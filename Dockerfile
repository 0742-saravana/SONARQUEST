# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create the model_weights directory
RUN mkdir -p backend/model_weights

# Download the ONNX model directly
ADD "https://github.com/0742-saravana/SONARQUEST/releases/download/V2.0/marineguardv2.onnx" backend/model_weights/marineguardv2.onnx

# Expose port 8080 for Cloud Run / Render
EXPOSE 8080

# Run the FastAPI server using the PORT environment variable
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
