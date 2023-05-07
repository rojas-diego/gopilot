# GoPilot

GoPilot is a small language model trained exclusively on Go code.

## Installation

Install the necessary dependencies using `conda` and the provided `environment.yml`

## Usage

**Process the dataset**: `python scripts/preprocess.py --help`

**Train the tokenizer**: `python scripts/train_tokenizer.py --help`

**Train the model**: `python scripts/train.py --help`

**Run inference on a trained model**: `python scripts/predict.py --help`

## Repository Structure

| Location          | Description                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `config`          | Stores sample model configuration files and tokenizer configuration files                                           |
| `flame`           | Python library to simplify training deep learning models using PyTorch                                              |
| `gopilot`         | Tokenizer, model and dataloader definitions for Gopilot                                                             |
| `scripts`         | Python scripts used to train the model, pre-process the dataset, train the tokenizer and run inference on the model |
| `environment.yml` | List of dependencies. Install with `conda` or `mamba`.                                                              |
