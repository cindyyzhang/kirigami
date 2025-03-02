# Kirigami Tiles

## Running with Docker and GPU Support

This project can be run on a GPU using Docker with NVIDIA Container Toolkit.

### Prerequisites

- NVIDIA GPU (tested with RTX 3090)
- Docker installed
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed

### Option 1: Using Docker directly

#### Building the Docker image

```bash
# Clone the repository if you haven't already
git clone <your-repository-url>
cd kirigami

# Build the Docker image
docker build -t kirigami:gpu .
```

#### Running the container with GPU support

```bash
# Run the container with GPU access
docker run --gpus all -it kirigami:gpu

# To run with interactive shell
docker run --gpus all -it kirigami:gpu /bin/bash

# To run with a mounted directory for output files
docker run --gpus all -v $(pwd)/output:/app/output -it kirigami:gpu
```

### Option 2: Using Docker Compose (Recommended)

This project includes a `docker-compose.yml` file for easier container management.

```bash
# Start the container in interactive mode
docker-compose up -d
docker-compose exec kirigami bash

# Or in a single command
docker-compose run kirigami

# To stop the container when done
docker-compose down
```

### Verifying GPU access inside the container

Once inside the container, you can verify that JAX can see the GPU by running:

```bash
# Run the GPU check script
python check_gpu.py

# Or directly in Python
python -c "import jax; print(jax.devices())"
```

This should show your NVIDIA GPU in the list of available devices.

### Running your code

Once inside the container with GPU access confirmed, you can run the main script:

```bash
python main.py
```

You can also use IPython or Jupyter for interactive development:

```bash
ipython
# or
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

