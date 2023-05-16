import logging
import os
import random
import string
from typing import List, Optional
import neptune
from neptune.utils import stringify_unsupported

from .utils import Metric


def neptune_is_available():
    return ("NEPTUNE_API_TOKEN" in os.environ)


class NeptuneTracker:
    """
    Responsible for reporting the status of a run to Neptune.ai.
    Tracks multiple metrics such as hyperparameters, losses, accuracies, etc.
    """

    def __init__(self, project: Optional[str] = None, api_token: Optional[str] = None):
        self.run = neptune.init_run(
            project=project if project is not None else self._getenv("NEPTUNE_PROJECT"),
            api_token=api_token if api_token is not None else self._getenv("NEPTUNE_API_TOKEN"),
            capture_stderr=True,
            capture_stdout=True,
            source_files=["./**/*.py"])
        logging.info(f"Run ID: {self.get_run_id()}")

    def _getenv(self, key: str):
        value = os.environ.get(key)
        if value is None:
            raise ValueError(f"Missing Neptune env key {key}. Please set it or provide it in the constructor of the tracker.")
        return value

    def track_hyperparameters(self, hyperparameters: dict):
        self.run["hyperparameters"] = stringify_unsupported(hyperparameters)

    def track_metrics(self, metrics: List[Metric]):
        for metric in metrics:
            self.run[metric.name].append(metric.value)

    def track_values(self, metrics: List[Metric]):
        for metric in metrics:
            self.run[metric.name] = metric.value

    def get_run_id(self):
        return self.run["sys/id"].fetch()


class NoopTracker:
    """
    A no-op tracker that does nothing.
    """

    def __init__(self):
        self.run_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))

    def track_hyperparameters(self, hyperparameters: dict):
        pass

    def track_metrics(self, metrics: List[Metric]):
        pass

    def track_values(self, metrics: List[Metric]):
        pass

    def get_run_id(self):
        return self.run_id
