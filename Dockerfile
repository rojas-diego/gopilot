FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"

# Install some necessary dependencies
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
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
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install the necessary python packages
RUN pip install --upgrade pip
RUN pip install datasets==2.12.0 tokenizers==0.13.3 transformers==4.29.1 pandas==2.0.1 pyarrow==10.0.1 neptune==1.0.2 tqdm==4.65.0 boto3==1.26.131

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
COPY ./tokenizer ./tokenizer
COPY ./model/ ./model
COPY ./flame/ ./flame
COPY ./*.py ./

# Build the shared library
RUN go build -o tokenizer/libgotok.so -buildmode=c-shared ./tokenizer/libgotok.go

# Triton leaks 32 bytes on every kernel invocation. The 2.0.0 release has been
# amended to fix this issue hence we must uninstall and reinstall the package.
# See https://github.com/pytorch/pytorch/issues/96937
RUN pip uninstall -y triton
RUN pip install triton==2.0.0.post1
