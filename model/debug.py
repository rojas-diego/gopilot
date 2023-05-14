import logging
import os
import time

import torch

from tokenizer import Tokenizer


class TrainingSampler:
    """
    At regular intervals during training, this class will sample from the
    model's predictions and record the input to a file. This helps us see how
    the model is progressing, what patterns it sees and whether the inputs and
    targets are correctly formed.
    """
    def __init__(self, output_dir: str, tokenizer: Tokenizer, max_batch_interval: int = 32, max_time_interval_secs: int = 32, max_files: int = 32):
        self.output_dir = output_dir
        self.max_batch_interval = max_batch_interval
        self.max_time_interval_secs = max_time_interval_secs
        self.max_files = max_files
        self.last_sample_ts = time.time()
        self.last_sample_batch = 0
        self.files_written = []
        self.tokenizer = tokenizer
        self.num_files_written = 0
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Training sampler enabled. Samples will be written to {self.output_dir}")

    def feed(self, inputs: torch.Tensor, targets: torch.Tensor, outputs: torch.Tensor):
        self.last_sample_batch += 1
        if self.last_sample_batch >= self.max_batch_interval or time.time() - self.last_sample_ts >= self.max_time_interval_secs:
            # Make a copy of these tensors on the CPU.
            inputs = inputs.detach().cpu()
            targets = targets.detach().cpu()
            outputs = outputs.detach().cpu().argmax(dim=-1)

            # Pick a random index in the batch
            idx = torch.randint(0, inputs.shape[0], (1,)).item()

            # Detokenize the inputs, targets and outputs
            sample_input_sequence = metaspace_cleanup(self.tokenizer.decode(inputs[idx,:].tolist()))
            sample_target_sequence = metaspace_cleanup(self.tokenizer.decode(targets[idx,:].tolist()))
            sample_output_sequence = metaspace_cleanup(self.tokenizer.decode(outputs[idx,:].tolist()))

            self.last_sample_batch = 0
            self.last_sample_ts = time.time()

            # Write the inputs, targets and outputs to a file
            filename = f'sample-{self.num_files_written:06d}.txt'
            location = os.path.join(self.output_dir, filename)
            with open(location, 'w') as f:
                f.write("--- INPUT ----\n")
                f.write(sample_input_sequence)
                f.write("\n--- TARGET ---\n")
                f.write(sample_target_sequence)
                f.write("\n--- OUTPUT ---\n")
                f.write(sample_output_sequence)
            self.files_written.append(location)
            self.num_files_written += 1

            # If we've reached the maximum number of files, delete the oldest.
            if len(self.files_written) > self.max_files:
                file = self.files_written.pop(0)
                if os.path.exists(file):
                    os.remove(file)

            logging.info(f'Wrote sample debug information to \'{location}\'')
