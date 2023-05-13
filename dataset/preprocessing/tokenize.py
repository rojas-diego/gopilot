import os
import pandas
from .base import PreprocessingJob
from tokenizer.tokenizer import HFTokenizer, GoScannerTokenizer, GoAstTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE

class TokenizeJob(PreprocessingJob):
    def run(self):
        for file in self.files():
            df = pandas.read_parquet(file)
            df["tokens"] = df["content"].apply(lambda x: self.tokenizer.encode(x)) # type: ignore
            self.save_parquet(df, os.path.basename(file))

class HFTokenizeJob(PreprocessingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = HFTokenizer(tokenizer=Tokenizer(model=BPE())) # type: ignore

class GoScannerTokenizeJob(PreprocessingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = GoScannerTokenizer()

class GoAstTokenizeJob(PreprocessingJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = GoAstTokenizer() # type: ignore
