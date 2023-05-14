# Handles training of the model over the provided dataset.

import argparse
import logging
import os
import sys

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import flame
from dataset import StreamParquetLoader
from model import GopilotModel, GopilotTask

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # General arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to the tokenizer configuration file.')
    args, remaining_args = parser.parse_known_args()
    # Training parameters
    tp_parser = argparse.ArgumentParser()
    tp_parser.add_argument('--gradient-accumulation-steps', type=int, help='Number of gradient accumulation steps (Default 1, no accumulation).')
    tp_parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    tp_parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability.')
    tp_parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay value.')
    tp_parser.add_argument('--warmup', type=int, default=100, help='Number of warmup steps.')
    tp_parser.add_argument('--lr', type=float, default=1e-3, help='Maximum learning rate.')
    tp_parser.add_argument('--epsilon', type=float, default=10e-12, help='AdamW epsilon parameter.')
    tp_parser.add_argument('--training-budget-secs', type=int, default=60*60, help='Training budget in seconds to define the learning rate schedule (Default 1h).')
    tp_parser.add_argument('--clip-gradients', type=float, default=0.5, help='Clip gradients norm value.')
    tp_parser.add_argument('--dataset', type=str, required=True, help='Location of the remote dataset.')
    tp_args, remaining_args = tp_parser.parse_known_args(remaining_args)
    # S3 arguments
    s3_parser = argparse.ArgumentParser()
    s3_parser.add_argument('--s3-bucket', type=str, default="gopilot", help='S3 bucket name.')
    s3_parser.add_argument('--s3-region', type=str, default="ap-east-1", help="S3 bucket region.")
    s3_parser.add_argument('--s3-cache-dir', type=str, default=".cache", help="Local cache directory for S3 files.")
    s3_args, remaining_args = s3_parser.parse_known_args(remaining_args)
    # Run arguments
    run_parser = argparse.ArgumentParser()
    run_parser.add_argument('--device', type=str, default='auto', help='Device to use for training.')
    run_parser.add_argument('--progress', default=False, action='store_true', help='Enable progress bar.')
    run_parser.add_argument('--verbose', default=True, action='store_true', help='Enable verbose logging.')
    run_parser.add_argument('--neptune', default=False, action='store_true', help='Enable Neptune integration.')
    run_parser.add_argument('--compile', default=False, action='store_true', help='Enable torch.compile().')
    run_args = run_parser.parse_args(remaining_args)
    
    # Transform args
    run_args.device = flame.best_device() if args.device == "auto" else torch.device(args.device)
    run_args.gradient_accumulation_steps = 1 if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps

    # Load the model
    model = GopilotModel.from_config_file(args.model, dropout=tp_args.dropout)
    flame.log_model_summary(model)

    # Optionally compile model
    if args.compile:
        assert args.device.type == "cuda", f"torch.compile() with Triton backend only runs on CUDA compatible devices."
        logging.info("Compiling model using 'inductor' backend")
        model: gpmodel.GPTModel = torch.compile(model, backend="inductor") # type: ignore

    # Load the dataset
    dataset = StreamParquetLoader(s3_args.bucket, s3_args.region, s3_args.cache_dir, args.prefix, args.batch_size)

    # Configure the tracker
    tracker = flame.NeptuneTracker("rojasdiegopro/gopilot") if (flame.neptune_is_available() and args.neptune) else flame.NoopTracker()
    tracker.track_hyperparameters(vars(tp_args))
    tracker.track_hyperparameters(model.hyperparams())

    # Configure optimizer, learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=args.epsilon) # TODO: betas should be made configurable
    scheduler = LambdaLR(optimizer, lr_lambda=flame.LinearLRScheduleWithTimeBudget(args.warmup, args.training_budget_secs, 0.1)) # TODO: better learning rate schedule
    criterion = CrossEntropyLoss()

    # Configure trainer
    trainer = flame.Trainer(GopilotTask(model, criterion, optimizer, scheduler, clip_gradients=args.clip_gradients), args.device)
    trainer.register_handlers(
        # Checkpoint every 1024 steps or every 30 minutes, whichever comes first
        flame.CheckpointingHandler(args.checkpoints_dir, filename_prefix=tracker.get_run_id()+"-step={step}-loss={loss:.2f}", max_step_interval=1024, max_time_interval_sec=60*30),
        flame.LoggingHandler(on_step=args.verbose, on_batch=False),
        flame.TrackingHandler(tracker, on_batch=False),
    )

    # Run training
    trainer.train(
        num_epochs=-1,
        train_loader=dataset,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        enable_progress_bar=args.progress,
    )
