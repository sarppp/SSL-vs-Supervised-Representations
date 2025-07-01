# Base image with CUDA and cuDNN
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Avoid user prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    curl wget git build-essential unzip \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install rclone
RUN curl https://rclone.org/install.sh | bash

# Symlink python3 and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Install uv (super-fast package manager)
RUN pip install uv

# Optional: upgrade pip
RUN pip install --upgrade pip

# Install ML libraries
# Install ML libraries
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir pandas numpy scikit-learn matplotlib seaborn jupyterlab notebook ipywidgets tqdm transformers datasets opencv-python accelerate

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]