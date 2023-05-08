# Handles training of the model over the provided dataset.
#
# This script expects a model configuration file to be provided, which specifies
# the hyperparameters of the model. The configuration file is a YAML file.
# See the `config` folder for examples.
#
# This script also expects a tokenizer configuration file, which specifies the
# learned tokenizer vocabulary. The file is passed to HuggingFace
# Tokenizer.from_file().

import argparse
import glob
import logging
import os
import sys
import time

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, schedule

# Add the parent directory of this script to the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flame

import gopilot.tokenizer as gptok
import gopilot.model as gpmodel
import gopilot.loader as gploader
import gopilot.task as gptask
import gopilot.debug as gpdebug

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to the tokenizer configuration file.')
    parser.add_argument('--dataset-dir', type=str, default="data/processed", help='Path to the directory containing Parquet dataset files.')
    parser.add_argument('--checkpoints-dir', type=str, default="checkpoints", help='Path where to store the checkpoint files.')
    # Training hyperparameters
    parser.add_argument('--gradient-accumulation-steps', type=int, help='Number of gradient accumulation steps (Default 1, no accumulation).')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability.')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay value.')
    parser.add_argument('--warmup-steps', type=int, default=100, help='Number of warmup steps.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Maximum learning rate.')
    parser.add_argument('--epsilon', type=float, default=10e-12, help='AdamW epsilon paramater.')
    parser.add_argument('--training-budget-secs', type=int, default=60*60, help='Training budget in seconds to define the learning rate schedule (Default 1h).')
    parser.add_argument('--clip-gradients', type=float, default=0.5, help='Clip gradients norm value.')
    # Training config
    parser.add_argument('--device', type=str, default="auto", help='Device to use for training.')
    parser.add_argument('--progress', default=False, action='store_true', help='Enable progress bar.')
    parser.add_argument('--verbose', default=True, action='store_true', help='Enable verbose logging.')
    parser.add_argument('--neptune', default=False, action='store_true', help='Enable Neptune integration.')
    parser.add_argument('--sampling', default=False, action='store_true', help='Enable debugging sampler.')
    parser.add_argument('--profiling', default=False, action='store_true', help='Enable profiling.')
    args = parser.parse_args()

    # Transform args
    args.device = flame.best_device() if args.device == "auto" else args.device
    args.gradient_accumulation_steps = 1 if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps

    # Load the model and tokenizer
    model = gpmodel.GPTModel.from_config_file(args.model, dropout=args.dropout)
    flame.xavier_initialization(model)
    flame.log_model_summary(model)
    tokenizer = gptok.load_tokenizer(args.tokenizer)
    assert tokenizer.get_vocab_size() == model.vocab_size, "The tokenizer vocabulary size does not match the model's vocabulary size."

    # Load the dataset
    dataset = gploader.GopilotDataset(
        files=glob.glob(os.path.join(args.dataset_dir, '*.parquet')),
        tokenizer=tokenizer,
        sequence_length=model.context_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)

    # Configure the tracker
    tracker = flame.NoopTracker()
    if flame.neptune_is_available() and args.neptune:
        tracker = flame.NeptuneTracker("rojasdiegopro/gopilot")
    logging.info(f"Run ID: {tracker.get_run_id()}")

    # Record the training config and model hyperparameters
    tracker.track_hyperparameters(vars(args))
    tracker.track_hyperparameters(model.hyperparams())

    # Configure optimizer, learning rate scheduler
    start_time = time.time()
    def learning_rate_scheduling(step):
        # This is really funky, should be improved
        if step < args.warmup_steps:
            return step / args.warmup_steps
        elapsed_time = time.time() - start_time
        scale_factor = 1 - (elapsed_time / args.training_budget_secs)
        if scale_factor < 0:
            return 0.1
        return max(0.1, scale_factor)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=args.epsilon) # betas should be made configurable
    criterion = CrossEntropyLoss()
    scheduler = LambdaLR(optimizer, lr_lambda=learning_rate_scheduling)

    # Configure debugging sampler
    sampler = gpdebug.TrainingSampler("samples/debug", tokenizer, max_batch_interval=64, max_time_interval_secs=60, max_files=10) if args.sampling else None

    # Configure trainer
    trainer = flame.Trainer(gptask.GopilotTask(model, criterion, optimizer, scheduler, clip_gradients=args.clip_gradients, sampler=sampler), args.device)
    trainer.register_handlers(
        # Checkpoint every 64 steps or every 30 minutes, whichever comes first
        flame.CheckpointingHandler(args.checkpoints_dir, filename_prefix=tracker.get_run_id()+"-step={step}-loss={loss:.2f}", max_step_interval=64, max_time_interval_sec=60*30),
        flame.LoggingHandler(on_step=args.verbose, on_batch=False),
        flame.TrackingHandler(tracker, on_batch=False),
    )

    # TODO: Add CUDA profiling, add better profiling support (regions)
    if args.profiling:
        trainer.register_handlers(flame.TorchProfilingHandler("profile.json", activities=[ProfilerActivity.CPU], schedule=schedule(wait=0, warmup=15, active=3, repeat=2)))

    # Run training
    trainer.train(
        # Train forever
        num_epochs=-1,
        train_loader=loader,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        enable_progress_bar=args.progress,
    )
