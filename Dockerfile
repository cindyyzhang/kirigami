FROM nvcr.io/nvidia/jax:23.12-py3

# Set working directory
WORKDIR /app

# (Optional) Update pip if needed.
RUN pip install --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    matplotlib \
    opencv-python \
    torch \
    torchvision \
    jaxopt \
    numpy \
    sympy

# Copy repository code
COPY . /app/

# Set environment variables for GPU
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Set a default command
CMD ["python", "main.py"] 