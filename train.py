# Handles training of the model over the provided dataset.

import argparse
import logging
import random
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import flame
from dataset import DistributedGopilotDataset
from model import GopilotModel, GopilotTask
from tokenizer import GopilotTokenizer, HuggingFaceTokenizer


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
    lr: float
    epsilon: float
    # Unfortunately, neptune.ai does not support recording values greater than
    # int32.max, so we have to use float instead.
    token_budget: float
    clip_gradients: float
    precision: Union[str, torch.dtype]
    warmup: int
    seed: int

@dataclass
class S3Args:
    s3_bucket: str
    s3_cache_dir: str
    s3_checkpoints: bool
    s3_dataset_prefix: str

@dataclass
class RunArgs:
    device: Union[str, torch.device]
    verbose: bool
    neptune: bool
    compile: bool
    checkpoints_dir: str

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # General arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-cf', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--tokenizer-cf', type=str, required=True, help='Path to the tokenizer configuration file.')
    parser.add_argument('--model', type=str, default="Gopilot", help='Name of the model to use.', choices=["gopilot"])
    parser.add_argument('--tokenizer', type=str, default="Gopilot", help='Name of the tokenizer to use.', choices=["gopilot", "hugging-face"])
    args, remaining_args = parser.parse_known_args()
    # Training parameters
    tp_parser = argparse.ArgumentParser()
    tp_parser.add_argument('--gradient-accumulation-steps', type=int, default=48, help='Number of gradient accumulation steps (Default 1, no accumulation).')
    tp_parser.add_argument('--batch-size', type=int, default=12, help='Batch size.')
    tp_parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability.')
    tp_parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay value.')
    tp_parser.add_argument('--lr', type=float, default=3e-4, help='Maximum learning rate.')
    tp_parser.add_argument('--epsilon', type=float, default=10e-12, help='AdamW epsilon parameter.')
    tp_parser.add_argument('--token-budget', type=float, default=1e10, help='Training budget in number of tokens to be processed.')
    tp_parser.add_argument('--clip-gradients', type=float, default=0.5, help='Clip gradients norm value.')
    tp_parser.add_argument('--precision', type=str, default="fp16", choices=["fp32", "fp16"], help='Precision to use for training.')
    tp_parser.add_argument('--seed', type=int, default=999, help='Random seed.')
    tp_parser.add_argument('--warmup', type=int, default=1000, help='Number of warmup steps.')
    tp_args, remaining_args = tp_parser.parse_known_args(remaining_args)
    # S3 arguments
    s3_parser = argparse.ArgumentParser()
    s3_parser.add_argument('--s3-bucket', type=str, default="gopilot", help='S3 bucket name.')
    s3_parser.add_argument('--s3-cache-dir', type=str, default=".cache", help='Local cache directory.')
    s3_parser.add_argument('--s3-checkpoints', default=False, action='store_true', help='Enable remote checkpoints.')
    s3_parser.add_argument('--s3-dataset-prefix', type=str, required=True, help='Prefix of the remote dataset.')
    s3_args, remaining_args = s3_parser.parse_known_args(remaining_args)
    # Run arguments
    run_parser = argparse.ArgumentParser()
    run_parser.add_argument('--device', type=str, default='auto', help='Device to use for training.')
    run_parser.add_argument('--verbose', default=True, action='store_true', help='Enable verbose logging.')
    run_parser.add_argument('--neptune', default=False, action='store_true', help='Enable Neptune integration.')
    run_parser.add_argument('--compile', default=False, action='store_true', help='Enable torch.compile().')
    run_parser.add_argument('--checkpoints-dir', type=str, default="out/checkpoints", help='Checkpoints directory.')
    run_args = run_parser.parse_args(remaining_args)

    args = Args(**vars(args))
    tp_args = TrainingParametersArgs(**vars(tp_args))
    s3_args = S3Args(**vars(s3_args))
    run_args = RunArgs(**vars(run_args))

    assert flame.s3_is_available(), "S3 is not available. Please set the relevant environment variables."
    assert flame.neptune_is_available() or not run_args.neptune, "Neptune is not available. Please set the relevant environment variables."

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
    if args.tokenizer == "gopilot":
        tokenizer = GopilotTokenizer.from_file(args.tokenizer_cf)
    else:
        tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer_cf)

    # Configure the tracker
    tracker = flame.NeptuneTracker("rojasdiegopro/gopilot") if (flame.neptune_is_available() and run_args.neptune) else flame.NoopTracker()
    tracker.track_hyperparameters(vars(args))
    tracker.track_hyperparameters(vars(tp_args))
    tracker.track_hyperparameters(vars(model.get_config()))
    tracker.track_hyperparameters({"dataset": s3_args.s3_dataset_prefix})

    # Load the dataset
    dataset = DistributedGopilotDataset(
        s3_args.s3_bucket,
        s3_args.s3_dataset_prefix,
        s3_args.s3_cache_dir,
        window_size=model.get_config().context_length+1,
        stride=model.get_config().context_length,
        rank=0,
        world_size=1,
    )
    loader = DataLoader(dataset, batch_size=tp_args.batch_size, num_workers=1, drop_last=True, pin_memory=run_args.device.type == "cuda", pin_memory_device="cuda" if run_args.device.type == "cuda" else "", prefetch_factor=32)

    # Configure optimizer, learning rate scheduler
    num_tokens_ingested_per_batch = tp_args.batch_size * tp_args.gradient_accumulation_steps * (model.get_config().context_length)
    total_steps = int(tp_args.token_budget) // num_tokens_ingested_per_batch
    logging.info(f"Compute budget summary: {tp_args.token_budget} tokens, {num_tokens_ingested_per_batch} tokens batch size, {total_steps} total steps, {flame.expected_loss(flame.model_size(model), tp_args.token_budget):.2f} expected loss.")
    optimizer = AdamW(model.parameters(), lr=tp_args.lr, weight_decay=tp_args.weight_decay, betas=(0.9, 0.999), eps=tp_args.epsilon) # TODO: betas should be made configurable
    scheduler = OneCycleLR(optimizer, max_lr=tp_args.lr, total_steps=total_steps, anneal_strategy='cos', pct_start=(tp_args.warmup/total_steps))

    # Configure trainer
    trainer = flame.Trainer(GopilotTask(model, optimizer, tokenizer.special_token_to_id("[PAD]"), scheduler, clip_gradients=tp_args.clip_gradients, precision=tp_args.precision), run_args.device)
    trainer.register_handlers(
        flame.CheckpointingHandler(
            run_args.checkpoints_dir,
            filename=tracker.get_run_id()+"-step={step}-loss={loss:.2f}.pt",
            max_files=3,
            max_step_interval=4096,
            max_time_interval_sec=60*60*2,
        ),
        flame.LoggingHandler(on_step=run_args.verbose, on_batch=False),
        flame.TrackingHandler(tracker),
        flame.S3RemoteCheckpointingHandler(
            s3_args.s3_bucket,
            f"checkpoints/{tracker.get_run_id()}",
            max_files=3
        ) if s3_args.s3_checkpoints else flame.NoopHandler(),
    )

    # Run training
    trainer.train(
        num_epochs=-1,
        train_loader=loader,
        gradient_accumulation_steps=tp_args.gradient_accumulation_steps,
    )
