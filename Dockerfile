FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install some necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up the environment for conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# Install Miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -p $CONDA_DIR -b && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Update conda itself
RUN conda update -n base conda

# Copy your environment.yml file
COPY environment.yml .

# Install the dependencies from the environment.yml file
RUN conda env update --name base --file environment.yml && \
    conda clean --all --yes

# Set the working directory
WORKDIR /workspace

# Copy your project files (you might want to specify specific files or directories)
COPY environment.yml .

# Copy your project files (specific files or directories)
COPY ./scripts/ ./scripts
COPY ./gopilot/ ./gopilot
COPY ./flame/ ./flame
COPY ./config/ ./config
