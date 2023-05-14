import os

import pandas

from tokenizer import GoScannerTokenizer, HuggingFaceTokenizer

from .base import PreprocessingJob


class TokenizationJob(PreprocessingJob):
    def run(self):
        for file in self.files():
            df = pandas.read_parquet(file)
            df["tokens"] = df["content"].apply(lambda x: self.tokenizer.encode(x)) # type: ignore
            self.save_parquet(df, os.path.basename(file))


class TokenizeWithHuggingFaceJob(TokenizationJob):
    def run(self):
        self.tokenizer = HuggingFaceTokenizer()
        super().run()


class TokenizeWithGoScannerJob(TokenizationJob):
    def run(self):
        self.tokenizer = GoScannerTokenizer()
        super().run()
