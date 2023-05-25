import datasets
import pandas
import tqdm

from .preprocessing_job import PreprocessingJob


def num_byte_to_human_readable(num_bytes: int) -> str:
    if num_bytes < 1000:
        return f"{num_bytes} B"
    elif num_bytes < 1000**2:
        return f"{num_bytes / 1000} KB"
    elif num_bytes < 1000**3:
        return f"{num_bytes / 1000**2} MB"
    elif num_bytes < 1000**4:
        return f"{num_bytes / 1000**3} GB"
    else:
        return f"{num_bytes / 1000**4} TB"


class UploadTheStackJob(PreprocessingJob):
    def run(self):
        super().run()
        the_stack_dedup_go = datasets.load_dataset('bigcode/the-stack-dedup', data_dir="data/go", split="train", use_auth_token=True)

        num_samples_per_shard = 64000

        current_shard_idx = 0
        current_shard_samples = []
        num_bytes_in_current_shard = 0
        
        progress_bar = tqdm.tqdm(the_stack_dedup_go, desc="Building shards", mininterval=1, leave=False)
        for sample in progress_bar:
            # Accumulate samples and bytes until we reach the shard size.
            num_bytes_in_current_shard += len(sample['content']) # type: ignore
            current_shard_samples.append(sample)

            # If we've reached the shard size, upload the shard and reset the
            # counters.
            if len(current_shard_samples) >= num_samples_per_shard:
                self.save_parquet(pandas.DataFrame(current_shard_samples), f"shard-{current_shard_idx:03}.parquet")
                current_shard_idx += 1
                current_shard_samples = []
                num_bytes_in_current_shard = 0

            # Update the progress bar.
            progress_bar.set_postfix_str(f"Current Shard Index: {current_shard_idx}, Size: {num_byte_to_human_readable(num_bytes_in_current_shard)}, Num Samples: {len(current_shard_samples)}", refresh=False)

        # If there are any samples left over, upload them as a final shard.
        if len(current_shard_samples) > 0:
            self.save_parquet(pandas.DataFrame(current_shard_samples), f"shard-{current_shard_idx:03}.parquet")
