# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Copy requirements file
COPY requirement.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy FastAPI application
COPY serve.py .

# Copy frontend files
COPY static ./static

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]