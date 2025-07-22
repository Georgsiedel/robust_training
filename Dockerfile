# Start from NVIDIA's CUDA image with cuDNN and Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-venv \
    git \
    && apt-get clean

# Create and activate a virtual environment for Python 3.11
RUN python3.11 -m venv /opt/venv

# Ensure the virtual environment is used for all Python commands
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /workspace

# Install Python dependencies (requirements.txt should be in your repo)
COPY requirements.txt /workspace/
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0
RUN pip install -r /workspace/requirements.txt

# Copy your repository code into the container
COPY . /workspace/

# Command to run your app
CMD ["bash"]
