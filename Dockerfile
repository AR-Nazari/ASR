# Use the CUDA base image with Ubuntu 24.04 and CUDA 12.6.2
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

# Set environment variables to prevent Python from writing .pyc files and to output logs immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python, pip, and python3-venv to create a virtual environment
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment inside the container
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY pip.conf /etc/pip.conf

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install Python dependencies from requirements.txt inside the new virtual environment
RUN pip install --no-cache-dir --default-timeout=50000 -r requirements.txt
RUN rm -fr requirements.txt /etc/pip.conf

# Expose the port Uvicorn will run on
EXPOSE 8080

# Run the application with Uvicorn
CMD ["python3", "main.py"]