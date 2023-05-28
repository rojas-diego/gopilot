# GoPilot

GoPilot is a small language model trained exclusively on Go code.

## Installation

You need to have `conda` and `go` installed on your machine. You can install the necessary dependencies using `conda` and the provided `environment_cpu.yml` (choose `environment_cuda.yml` when running CUDA).

Build the Go tokenizer binary:

```bash
# Linux, MacOS
go build -o tokenizer/libgotok.so -buildmode=c-shared ./tokenizer/libgotok.go
# Windows
go build -o tokenizer/libgotok.dll -buildmode=c-shared ./tokenizer/libgotok.go
```

## Usage

### Pre-Training

A CUDA Docker image is made available. Here are the required parameters.

```bash
docker run \
    -d \
    --gpus '"device=0"' \
    --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --env AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION \
    --env NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc host \
    --net host \
    rojasdiego/gopilot:latest \
    python train.py \
        --model-cf model/config/gopilot-290M.yml \
        --tokenizer hugging-face \
        --tokenizer-cf tokenizer/config/hugging-face.json \
        --s3-dataset-prefix datasets/the-stack-dedup-v1.2/hugging-face-pretokenized \
        --gradient-accumulation-steps 64 \
        --batch-size 12 \
        --lr 0.0005 \
        --token-budget 20000000000 \
        --device cuda \
        --precision fp16 \
        --s3-checkpoints \
        --warmup 1000 \
        --neptune \
        --compile
```

### Inference Server

The inference server is a simple HTTP server that hosts the model and exposes a `/complete` endpoint to submit samples to auto-complete.

```
python inference_server.py --help
```

### VSCode Extension

Check out the Gopilot VSCode extension [here](https://github.com/rojas-diego/gopilot-vscode-ext). Works with the inference server.

## Repository Structure

| Location           | Description                                                                 |
| ------------------ | --------------------------------------------------------------------------- |
| `dataset`          | The gopilot Transformer model and objectives                                |
| `model`            | Implementation of a Go specific tokenizer                                   |
| `tokenizer`        | Processing jobs and utilities for the dataset used for pre-training Gopilot |
| `flame`            | Python library to simplify training deep learning models using PyTorch      |
| `requirements.txt` | Production dependencies.                                                    |
| `environment.yml`  | Development dependencies. Install with `conda`.                             |
