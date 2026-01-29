# Splat to Mesh Pipeline
# Converts Gaussian Splat PLY files to Unity-ready meshes

FROM python:3.10-slim

# Install system dependencies required by Open3D
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline scripts
COPY splat_to_pointcloud.py .
COPY pointcloud_to_mesh.py .
COPY run_pipeline.py .

# Create directories for input/output
RUN mkdir -p /data/input /data/output

# Set the data directory as the working directory for running commands
WORKDIR /data

# Default command shows help
ENTRYPOINT ["python", "/app/run_pipeline.py"]
CMD ["--help"]
