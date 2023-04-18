import logging
import os
import random
import string
from typing import List
from dotenv import dotenv_values
import neptune
from neptune.utils import stringify_unsupported

from dlutils.utils import Metric



NEPTUNE_CONFIG_FILENAME = ".neptune-config"


def neptune_is_available():
    # Search for a .neptune-config file in the current directory.
    if os.path.isfile(f"{NEPTUNE_CONFIG_FILENAME}"):
        return True
    # Search for two environment variables NEPTUNE_API_TOKEN and NEPTUNE_PROJECT.
    return "NEPTUNE_API_TOKEN" in os.environ and "NEPTUNE_PROJECT" in os.environ


class NeptuneTracker:
    """
    Responsible for reporting the status of a run to Neptune.ai.
    Tracks multiple metrics such as hyperparameters, losses, accuracies, etc.
    """

    def __init__(self, with_id: str | None = None):
        api_token, project = self._get_neptune_credentials_from_environment()
        self.run = neptune.init_run(
            project=project,
            api_token=api_token,
            with_id=with_id,
            capture_stderr=False,
            capture_stdout=False,
            source_files=["*.py"])

    def _get_neptune_credentials_from_environment(self):
        # Load key-value pairs from the .env file into a dictionary
        env_dict = dotenv_values(NEPTUNE_CONFIG_FILENAME)

        api_token = env_dict.get("NEPTUNE_API_TOKEN", os.environ.get("NEPTUNE_API_TOKEN"))
        project = env_dict.get("NEPTUNE_PROJECT", os.environ.get("NEPTUNE_PROJECT"))

        if api_token is None or project is None:
            raise ValueError("No Neptune credentials found. Please create a {} file or set the environment variables NEPTUNE_API_TOKEN and NEPTUNE_PROJECT.".format(NEPTUNE_CONFIG_FILENAME))

        return api_token, project

    def track_hyperparameters(self, hyperparameters: dict):
        self.run["hyperparameters"] = stringify_unsupported(hyperparameters)

    def track_metrics(self, metrics: List[Metric]):
        for metric in metrics:
            self.run[metric.name] = metric.value

    def track_epoch(self, epoch: int):
        self.run["hyperparameters/epochs"] = epoch

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

    def track_epoch(self, epoch: int):
        pass

    def get_run_id(self):
        return self.run_id
