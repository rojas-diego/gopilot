# GoPilot

GoPilot is a small language model trained exclusively on Go code.

## Installation

You need to have `conda` and `go` installed on your machine. You can install the necessary dependencies using `conda` and the provided `environment_cpu.yml` (choose `environment_cuda.yml` when running CUDA).

Build the Go tokenizer binary:

```bash
# Linux, MacOS
go build -o gotok/libgotok.so -buildmode=c-shared ./gotok/tokenizer.go
# Windows
go build -o gotok/libgotok.dll -buildmode=c-shared ./gotok/tokenizer.go
```

## Usage

**Process the dataset**: `python scripts/preprocess.py --help`

**Train the tokenizer**: `python scripts/train_tokenizer.py --help`

**Train the model**: `python scripts/train.py --help`

**Run inference on a trained model**: `python scripts/predict.py --help`

## Repository Structure

| Location                      | Description                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `demo`                        | Nano text editor to showcase Gopilot                                                                                |
| `config`                      | Stores sample model configuration files and tokenizer configuration files                                           |
| `flame`                       | Python library to simplify training deep learning models using PyTorch                                              |
| `gopilot`                     | The gopilot Transformer model and objectives                                                                        |
| `gotok`                       | Implementation of a Go specific tokenizer                                                                           |
| `scripts`                     | Python scripts used to train the model, pre-process the dataset, train the tokenizer and run inference on the model |
| `environment_{cpu\|cuda}.yml` | List of dependencies. Install with `conda` or `mamba`.                                                              |
