import argparse
import atexit
import dataclasses
import logging
import os
import random
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import flame
from dataset import GopilotFineTuningDataset
from model import GopilotModel, GopilotTask
from tokenizer import GopilotTokenizer, HuggingFaceTokenizer, Tokenizer
from eval import evaluate_humanevalx_pass_at_k

@dataclasses.dataclass
class Args:
    model_cf: str
    tokenizer: str
    tokenizer_cf: str
    in_model_weights: str
    out_model_weights: str
    dataset_filepath: str

@dataclasses.dataclass
class TrainingParametersArgs:
    gradient_accumulation_steps: int
    batch_size: int
    dropout: float
    weight_decay: float
    lr: float
    epsilon: float
    num_epochs: int
    clip_gradients: float
    precision: Union[str, torch.dtype]
    seed: int

@dataclasses.dataclass
class S3Args:
    s3_bucket: str
    s3_cache_dir: str
    s3_checkpoints: bool

@dataclasses.dataclass
class RunArgs:
    device: Union[str, torch.device]
    verbose: bool
    neptune: bool


class EvaluateAtBeginningAndAfterEachEpochHandler:
    def __init__(self, model: GopilotModel, loaders: List[Tuple[str, DataLoader]], pad_token_id: int, tokenizer=None):
        self.model = model
        self.loaders = loaders
        self.pad_token_id = pad_token_id
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.tokenizer = tokenizer

    def on_train_start(self, trainer: flame.Trainer):
        trainer.task.eval(trainer.device)
        self._evaluate(trainer)
        trainer.task.train(trainer.device)

    def on_epoch_end(self, trainer: flame.Trainer, epoch_idx: int):
        trainer.task.eval(trainer.device)
        self._evaluate(trainer)
        trainer.task.train(trainer.device)

    def _evaluate(self, trainer: flame.Trainer):
        # Evaluate the validation loss on a portion of the fine-tuning dataset.
        with torch.no_grad():
            for (name, loader) in self.loaders:
                losses = []
                logging.info(f"Evaluating on '{name}'...")
                for batch_idx, batch in enumerate(loader):
                    batch = batch.to(trainer.device)
                    batch_size, sequence_length = batch.shape[0], batch.shape[1]-1
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]
                    attention_mask = (inputs != self.pad_token_id).long()
                    outputs = self.model(inputs, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss_mask = (targets != self.pad_token_id).float()
                    loss = self.criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
                    masked_loss = loss * loss_mask.view(-1)
                    total_loss = torch.sum(masked_loss)
                    num_active_elements = torch.sum(loss_mask)
                    if num_active_elements == 0:
                        num_active_elements = torch.tensor(1e-8)
                    loss = total_loss / num_active_elements
                    outputs.clear()
                    losses.append(loss.item())
                logging.info(f"Validation loss on '{name}': {np.mean(losses)}")
        # Evaluate HumanEvalX score.
        results = evaluate_humanevalx_pass_at_k(self.tokenizer, self.model, 100, 256, False)
        logging.info(f"HumanEvalX pass@100: {results['pass@100']}")
        logging.info(f"HumanEvalX compile@100: {results['compile@100']}")


KNOWN_FINE_TUNING_DATASETS = {
    "programs-from-descriptions": "dataset/finetuning/programs-from-descriptions.jsonl",
    "idiomatic-programs": "dataset/finetuning/idiomatic-programs.jsonl",
    "simple-functions": "dataset/finetuning/simple-functions.jsonl",
}


def determnistic_shuffle_and_split(filepath: str, tokenizer: Tokenizer, window_size: int, stride: int, seed: int, split: float):
    dataset = GopilotFineTuningDataset(filepath, tokenizer, window_size, stride)
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_idx = int(len(indices) * split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # General arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-cf', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--tokenizer-cf', type=str, required=True, help='Path to the tokenizer configuration file.')
    parser.add_argument('--tokenizer', type=str, default="hugging-face", help='Name of the tokenizer to use.', choices=["gopilot", "hugging-face"])
    parser.add_argument('--in-model-weights', type=str, default=None, help='Path to the model weights.')
    parser.add_argument('--out-model-weights', type=str, default=None, help='Path to which the fine-tuned model weights will be saved.')
    parser.add_argument('--dataset-filepath', type=str, default=None, help='Path to the JSONL dataset file. Can also specify a known dataset name. (e.g. "programs-from-descriptions")')
    args, remaining_args = parser.parse_known_args()
    # Training parameters
    tp_parser = argparse.ArgumentParser()
    tp_parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of gradient accumulation steps.')
    tp_parser.add_argument('--batch-size', type=int, default=1, help='Batch size.')
    tp_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability.')
    tp_parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay value.')
    tp_parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    tp_parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon value for AdamW.')
    tp_parser.add_argument('--num-epochs', type=int, default=1, help='Number of epochs.')
    tp_parser.add_argument('--clip-gradients', type=float, default=1.0, help='Clip gradients.')
    tp_parser.add_argument('--precision', type=str, default="float32", help='Precision.')
    tp_parser.add_argument('--seed', type=int, default=999, help='Random seed.')
    tp_args, remaining_args = tp_parser.parse_known_args(remaining_args)
    # S3 arguments
    s3_parser = argparse.ArgumentParser()
    s3_parser.add_argument('--s3-bucket', type=str, default=None, help='S3 bucket.')
    s3_parser.add_argument('--s3-cache-dir', type=str, default=None, help='S3 cache directory.')
    s3_parser.add_argument('--s3-checkpoints', action='store_true', help='Upload checkpoints to S3.')
    s3_args, remaining_args = s3_parser.parse_known_args(remaining_args)
    # Run arguments
    run_parser = argparse.ArgumentParser()
    run_parser.add_argument('--device', type=str, default="cuda", help='Device to use.', choices=["cpu", "cuda", "mps"])
    run_parser.add_argument('--verbose', action='store_true', default=True, help='Verbose.')
    run_parser.add_argument('--neptune', action='store_true', help='Log to Neptune.')
    run_args = run_parser.parse_args(remaining_args)

    # Parse args
    args = Args(**vars(args))
    tp_args = TrainingParametersArgs(**vars(tp_args))
    s3_args = S3Args(**vars(s3_args))
    run_args = RunArgs(**vars(run_args))

    # Check S3
    assert flame.s3_is_available(), "S3 is not available. Please set the relevant environment variables."

    # Seed for reproducibility
    torch.manual_seed(tp_args.seed)
    np.random.seed(tp_args.seed)
    random.seed(tp_args.seed)

    # Transform args
    run_args.device = flame.best_device() if run_args.device == "auto" else torch.device(run_args.device)
    tp_args.precision = torch.float32 if tp_args.precision == "fp32" else torch.float16

    # Model
    model = GopilotModel.from_config_file(args.model_cf, tp_args.dropout)
    
    # Load model from checkpoint
    checkpoint = torch.load(args.in_model_weights, map_location=run_args.device)
    for key in list(checkpoint['model'].keys()):
        if key.startswith("_orig_mod."):
            checkpoint['model'][key[len("_orig_mod."):]] = checkpoint['model'].pop(key)
    model.load_state_dict(checkpoint['model'])

    # Load the tokenizer
    if args.tokenizer == "gopilot":
        tokenizer = GopilotTokenizer.from_file(args.tokenizer_cf)
    else:
        tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer_cf)

    # Load the dataset
    validation_loaders = []
    if args.dataset_filepath in KNOWN_FINE_TUNING_DATASETS:
        logging.info(f"Loading known dataset {args.dataset_filepath} from {KNOWN_FINE_TUNING_DATASETS[args.dataset_filepath]}")
        train_ds, validation_ds = determnistic_shuffle_and_split(KNOWN_FINE_TUNING_DATASETS[args.dataset_filepath], tokenizer, model.get_config().context_length+1, model.get_config().context_length, tp_args.seed, 0.9)
    else:
        train_ds, validation_ds = determnistic_shuffle_and_split(args.dataset_filepath, tokenizer, model.get_config().context_length+1, model.get_config().context_length, tp_args.seed, 0.9)
    logging.info(f"Using {len(train_ds)} samples for training and {len(validation_ds)} samples for validation")
    validation_loaders.append((os.path.basename(args.dataset_filepath), DataLoader(validation_ds, batch_size=tp_args.batch_size, shuffle=False)))

    # Add all the known datasets as validation baselines.
    for dataset_name, dataset_filepath in KNOWN_FINE_TUNING_DATASETS.items():
        if args.dataset_filepath == dataset_name or dataset_filepath == args.dataset_filepath:
            continue
        ds = GopilotFineTuningDataset(dataset_filepath, tokenizer, model.get_config().context_length+1, model.get_config().context_length)
        logging.info(f"Using known dataset {dataset_filepath} as validation baseline ({len(ds)} samples)")
        validation_loaders.append((dataset_name, DataLoader(ds, batch_size=tp_args.batch_size, shuffle=False)))

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=tp_args.lr, eps=tp_args.epsilon, weight_decay=tp_args.weight_decay)

    tracker = flame.NeptuneTracker("rojasdiegopro/gopilot") if (flame.neptune_is_available() and run_args.neptune) else flame.NoopTracker()

    # Configure trainer
    trainer = flame.Trainer(
        GopilotTask(
            model, # type: ignore
            optimizer,
            pad_token_id=tokenizer.special_token_to_id("[PAD]"),
            clip_gradients=tp_args.clip_gradients,
            precision=tp_args.precision
        ),
        run_args.device)
    trainer.register_handlers(
        flame.LoggingHandler(on_step=run_args.verbose, on_batch=run_args.verbose),
        EvaluateAtBeginningAndAfterEachEpochHandler(
            model,
            validation_loaders,
            tokenizer.special_token_to_id("[PAD]"),
            tokenizer=tokenizer
        ),
        flame.TrackingHandler(tracker),
    )

    def save_weights_atexit():
        # Save model
        trainer.task.checkpoint(args.out_model_weights, 0, 0, 0, [])
        logging.info(f"Model saved to {args.out_model_weights}")

    atexit.register(save_weights_atexit)

    # Run training
    trainer.train(
        num_epochs=tp_args.num_epochs,
        train_loader=DataLoader(train_ds, batch_size=tp_args.batch_size, shuffle=True),
        gradient_accumulation_steps=tp_args.gradient_accumulation_steps,
    )

