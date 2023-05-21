import tempfile
import pandas

from tokenizer import GopilotTokenizer, HuggingFaceTokenizer, Trainer

from .preprocessing_job import PreprocessingJob


class TrainTokenizerJob(PreprocessingJob):
    def run(self):
        trainer: Trainer = self.tokenizer.new_trainer( # type: ignore
            vocab_size=2**15,
            special_tokens=["[UNK]", "[EOS]", "[PAD]"],
        )
        for file in self.files():
            dataset_shard = pandas.read_parquet(file)
            # For each batch of 16,000 samples, train the tokenizer
            accumulated = []
            for content in dataset_shard["content"].tolist():
                # Remove non-ascii characters from the dataset
                accumulated.append(content.encode("ascii", errors="ignore").decode())
                if len(accumulated) >= 16000:
                    trainer.train_from_iterator(accumulated)
                    accumulated = []
            if accumulated:
                trainer.train_from_iterator(accumulated) # type: ignore
            # Train on one chunk of the dataset for now
            break
        # Save the tokenizer
        with tempfile.NamedTemporaryFile() as f:
            self.tokenizer.save(f.name) # type: ignore
            self.save(f.name, "tokenizer.json")


class TrainHuggingFaceTokenizerJob(TrainTokenizerJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = HuggingFaceTokenizer()


class TrainGopilotTokenizerJob(TrainTokenizerJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = GopilotTokenizer()
