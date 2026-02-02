# Splat to Mesh Pipeline
# Converts Gaussian Splat PLY files to Unity-ready meshes using BPA

FROM python:3.10-slim

# Disable Python output buffering for real-time logs
ENV PYTHONUNBUFFERED=1

# Default configuration (can be overridden via docker-compose or docker run -e)
ENV INPUT_FILE=""
ENV OUTPUT_FILE=""
ENV OPACITY_THRESHOLD=0.3
ENV OUTLIER_STD_RATIO=2.0
ENV THIN_GEOMETRY=true
ENV FLIP_NORMALS=false
ENV DOUBLE_SIDED=false
ENV SMOOTH_FINAL=false
ENV SMOOTH_ITERATIONS=5
ENV KEEP_INTERMEDIATE=false
ENV VERBOSE=true

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

# Create data directory
RUN mkdir -p /data

# Set the data directory as working directory
WORKDIR /data

# Run the pipeline
CMD ["python", "/app/run_pipeline.py"]
