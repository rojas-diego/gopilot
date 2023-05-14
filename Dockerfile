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
COPY environment_cuda.yml .

# Install the dependencies from the environment.yml file
RUN conda env update --name base --file environment_cuda.yml && \
    conda clean --all --yes

# Install Golang
ENV GOLANG_VERSION 1.18.1
ENV GOPATH /go
ENV GOROOT /usr/local/go
ENV PATH $GOROOT/bin:$GOPATH/bin:$PATH

RUN wget -q https://dl.google.com/go/go${GOLANG_VERSION}.linux-amd64.tar.gz -O go.tar.gz && \
    tar -C /usr/local -xzf go.tar.gz && \
    rm go.tar.gz && \
    mkdir -p "$GOPATH/src" "$GOPATH/bin"

# Set the working directory
WORKDIR /workspace

# Copy your project files (specific files or directories)
COPY ./dataset ./dataset
COPY ./flame/ ./flame
COPY ./model/ ./model
COPY ./scripts/ ./scripts
COPY ./tests ./tests
COPY ./tokenizer/ ./tokenizer

# Build the shared library
RUN go build -o tokenizer/libgotok.so -buildmode=c-shared ./tokenizer/libgotok.go
