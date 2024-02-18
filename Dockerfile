FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install cpu version of torch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# First, copy only requirements.txt and install Python dependencies to leverage Docker cache
COPY requirements_docker.txt requirements.txt
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Then copy the rest of the application
COPY . .