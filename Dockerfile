# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install system dependencies (if needed for some LLM packages)
RUN apt-get update && apt-get install -y build-essential && apt-get clean

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose Gradio port
EXPOSE 7860

# Default command
CMD ["python", "app.py"]