from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional, Union

import pandas

import flame


class Extractor(ABC):
    """Extracts samples from the dataset."""
    @abstractmethod
    def samples(self, files: Iterable[str]) -> Iterable:
        pass


class ParquetExtractorWithTokenization(Extractor):
    def __init__(self, transform: Callable, shuffle: bool = True, tracker: Optional[Union[flame.NeptuneTracker, flame.NoopTracker]] = None):
        self.transform = transform
        self.shuffle = shuffle
        self.tracker = tracker
        self.rows_visited = 0

    def samples(self, files: Iterable[str]) -> Iterable[List[int]]:
        """Extracts a DataFrame from a Parquet file. Tokenizes the "content" column into a "tokens" column."""
        for filepath in files:
            df = pandas.read_parquet(filepath)
            if self.shuffle:
                df = df.sample(frac=1)
            for _, row in df.iterrows():
                self.rows_visited += 1
                if self.tracker:
                    self.tracker.track_values([flame.Metric("dataset/rows_visited", self.rows_visited)])
                yield self.transform(row["content"])
