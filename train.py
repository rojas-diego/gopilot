# Handles training of the model over the provided dataset.

import argparse
import logging
import os
import random
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import flame
from dataset import (CachedS3DataSource, DataPipeline, ParquetExtractorWithTokenization, VariableLengthStridedWindowBatcher)
from model import GopilotModel, GopilotTask
from tokenizer import GoScannerTokenizer, HuggingFaceTokenizer


@dataclass
class Args:
    model: str
    model_cf: str
    tokenizer: str
    tokenizer_cf: str

@dataclass
class TrainingParametersArgs:
    gradient_accumulation_steps: int
    batch_size: int
    dropout: float
    weight_decay: float
    warmup: int
    lr: float
    epsilon: float
    training_budget_secs: int
    clip_gradients: float
    precision: Union[str, torch.dtype]
    dataset: str
    seed: int

@dataclass
class S3Args:
    s3_bucket: str
    s3_cache_dir: str

@dataclass
class RunArgs:
    device: Union[str, torch.device]
    progress: bool
    verbose: bool
    neptune: bool
    compile: bool
    checkpoints_dir: str
    remote_checkpoints: bool

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # General arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-cf', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--tokenizer-cf', type=str, required=True, help='Path to the tokenizer configuration file.')
    parser.add_argument('--model', type=str, default="Gopilot", help='Name of the model to use.', choices=["Gopilot"])
    parser.add_argument('--tokenizer', type=str, default="GoScanner", help='Name of the tokenizer to use.', choices=["GoScanner", "HuggingFace"])
    args, remaining_args = parser.parse_known_args()
    # Training parameters
    tp_parser = argparse.ArgumentParser()
    tp_parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of gradient accumulation steps (Default 1, no accumulation).')
    tp_parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    tp_parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability.')
    tp_parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay value.')
    tp_parser.add_argument('--warmup', type=int, default=100, help='Number of warmup steps.')
    tp_parser.add_argument('--lr', type=float, default=1e-3, help='Maximum learning rate.')
    tp_parser.add_argument('--epsilon', type=float, default=10e-12, help='AdamW epsilon parameter.')
    tp_parser.add_argument('--training-budget-secs', type=int, default=60*60, help='Training budget in seconds to define the learning rate schedule (Default 1h).')
    tp_parser.add_argument('--clip-gradients', type=float, default=0.5, help='Clip gradients norm value.')
    tp_parser.add_argument('--precision', type=str, default="fp32", choices=["fp32", "fp16"], help='Precision to use for training.')
    tp_parser.add_argument('--dataset', type=str, required=True, help='Prefix of the remote dataset.')
    tp_parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    tp_args, remaining_args = tp_parser.parse_known_args(remaining_args)
    # S3 arguments
    s3_parser = argparse.ArgumentParser()
    s3_parser.add_argument('--s3-bucket', type=str, default="gopilot", help='S3 bucket name.')
    s3_parser.add_argument('--s3-cache-dir', type=str, default=".cache", help='Local cache directory.')
    s3_args, remaining_args = s3_parser.parse_known_args(remaining_args)
    # Run arguments
    run_parser = argparse.ArgumentParser()
    run_parser.add_argument('--device', type=str, default='auto', help='Device to use for training.')
    run_parser.add_argument('--progress', default=False, action='store_true', help='Enable progress bar.')
    run_parser.add_argument('--verbose', default=True, action='store_true', help='Enable verbose logging.')
    run_parser.add_argument('--neptune', default=False, action='store_true', help='Enable Neptune integration.')
    run_parser.add_argument('--compile', default=False, action='store_true', help='Enable torch.compile().')
    run_parser.add_argument('--checkpoints-dir', type=str, default="out/checkpoints", help='Checkpoints directory.')
    run_parser.add_argument('--remote-checkpoints', default=False, action='store_true', help='Enable remote checkpoints.')
    run_args = run_parser.parse_args(remaining_args)

    args = Args(**vars(args))
    tp_args = TrainingParametersArgs(**vars(tp_args))
    s3_args = S3Args(**vars(s3_args))
    run_args = RunArgs(**vars(run_args))

    assert "AWS_DEFAULT_REGION" in os.environ, "AWS_DEFAULT_REGION environment variable must be set."
    assert "AWS_ACCESS_KEY_ID" in os.environ, "AWS_ACCESS_KEY_ID environment variable must be set."
    assert "AWS_SECRET_ACCESS_KEY" in os.environ, "AWS_SECRET_ACCESS_KEY environment variable must be set."
    assert "NEPTUNE_API_TOKEN" in os.environ or not run_args.neptune, "NEPTUNE_API_TOKEN environment variable must be set for --neptune to work."

    # Seed for reproducibility
    torch.manual_seed(tp_args.seed)
    np.random.seed(tp_args.seed)
    random.seed(tp_args.seed)

    # Transform args
    run_args.device = flame.best_device() if run_args.device == "auto" else torch.device(run_args.device)
    tp_args.precision = torch.float32 if tp_args.precision == "fp32" else torch.float16

    # Load the model
    model = GopilotModel.from_config_file(args.model_cf, dropout=tp_args.dropout)
    flame.log_model_summary(model)

    # Optionally compile model
    if run_args.compile:
        assert run_args.device.type == "cuda", f"torch.compile() with Triton backend only runs on CUDA compatible devices."
        model: GopilotModel = torch.compile(model, backend="inductor") # type: ignore

    # Load the tokenizer
    if args.tokenizer == "GoScanner":
        tokenizer = GoScannerTokenizer.from_file(args.tokenizer_cf)
    else:
        tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer_cf)

    # Configure the tracker
    tracker = flame.NeptuneTracker("rojasdiegopro/gopilot") if (flame.neptune_is_available() and run_args.neptune) else flame.NoopTracker()
    tracker.track_hyperparameters(vars(args))
    tracker.track_hyperparameters(vars(tp_args))
    tracker.track_hyperparameters(vars(model.get_config()))

    # Load the dataset
    dataset = DataPipeline(
        CachedS3DataSource(s3_args.s3_bucket, s3_args.s3_cache_dir, tp_args.dataset, tracker=tracker),
        ParquetExtractorWithTokenization(tokenizer.encode, tracker=tracker),
        VariableLengthStridedWindowBatcher(tp_args.batch_size, model.get_config().context_length+1, tokenizer.special_token_to_id("[PAD]"), tokenizer.special_token_to_id("[EOS]"), stride_range=(model.get_config().context_length, model.get_config().context_length)),
    )

    # Configure optimizer, learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=tp_args.lr, weight_decay=tp_args.weight_decay, betas=(0.9, 0.98), eps=tp_args.epsilon) # TODO: betas should be made configurable
    scheduler = LambdaLR(optimizer, lr_lambda=flame.LinearLRScheduleWithTimeBudget(tp_args.warmup, tp_args.training_budget_secs, 0.1)) # TODO: better learning rate schedule
    criterion = CrossEntropyLoss()

    # Configure trainer
    trainer = flame.Trainer(GopilotTask(model, optimizer, tokenizer.special_token_to_id("[PAD]"), scheduler, clip_gradients=tp_args.clip_gradients, precision=tp_args.precision), run_args.device)
    trainer.register_handlers(
        flame.CheckpointingHandler(
            run_args.checkpoints_dir,
            filename_prefix=tracker.get_run_id()+"-step={step}-loss={loss:.2f}.pt",
            max_files=3,
            max_step_interval=2**15,
            max_time_interval_sec=60*60*2,
            # Optionally save the checkpoints to S3
            remote_bucket=s3_args.s3_bucket if run_args.remote_checkpoints else None,
            remote_prefix=f"checkpoints/{tracker.get_run_id()}" if run_args.remote_checkpoints else None,
        ),
        flame.LoggingHandler(on_step=run_args.verbose, on_batch=False),
        flame.TrackingHandler(tracker),
    )

    # Run training
    trainer.train(
        num_epochs=-1,
        train_loader=dataset,
        gradient_accumulation_steps=tp_args.gradient_accumulation_steps,
        enable_progress_bar=run_args.progress,
    )
