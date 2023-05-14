import logging
from abc import ABC, abstractmethod
import os
from typing import Callable, Iterable, List

import pandas


class Extractor(ABC):
    """Extracts samples from the dataset."""
    @abstractmethod
    def samples(self, files: Iterable[str]) -> Iterable:
        pass


class ParquetExtractorWithTokenization(Extractor):
    def __init__(self, transform: Callable, shuffle: bool = True):
        self.transform = transform
        self.shuffle = shuffle

    def samples(self, files: Iterable[str]) -> Iterable[List[int]]:
        """Extracts a DataFrame from a Parquet file. Tokenizes the "content" column into a "tokens" column."""
        for filepath in files:
            df = pandas.read_parquet(filepath)
            if self.shuffle:
                df = df.sample(frac=1)
            for index, row in df.iterrows():
                # file_name = ""
                # repo_name = ""
                # try:
                #     file_name = os.path.basename(row['max_stars_repo_path'])
                #     repo_name = row['max_stars_repo_name']
                # except Exception:
                #     pass
                # logging.info(f"Extracting '{file_name}' from '{repo_name}'")
                yield self.transform(row["content"])